#!/usr/bin/env python3
"""Convert a flat image folder, or one or more video files, into a COLMAP
dataset (images/ + database.db + sparse/).

Target layout, matching ~/Datasets/person-hall:

    <output>/
      images/         copied (or linked) source images, or frames sampled from the videos
      database.db     COLMAP feature/match database
      sparse/         reconstructed model in TXT format
        cameras.txt
        images.txt
        points3D.txt

Examples:
    ./lichtfeld_preparater.py ~/Datasets/belval/images_long1600 ~/Datasets/belval-colmap
    ./lichtfeld_preparater.py ~/Videos/hall.mp4 ~/Datasets/hall-colmap --fps 3
    ./lichtfeld_preparater.py ~/Videos/hall-1.mp4 ~/Videos/hall-2.mp4 ~/Datasets/hall-colmap
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help=__doc__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".mts", ".insv"}

AUTO_FPS = -1.0  # --fps sentinel: pick a rate from the video's duration
TARGET_FRAMES = 200  # frame count --fps auto aims for on a video of any length
MIN_AUTO_FPS = 1.0  # below this, consecutive frames stop overlapping enough to match
FALLBACK_FPS = 2.0  # used when ffprobe can't tell us the duration


class Matcher(str, Enum):
    exhaustive = "exhaustive"
    sequential = "sequential"
    vocab_tree = "vocab_tree"
    spatial = "spatial"


class CameraModel(str, Enum):
    simple_pinhole = "SIMPLE_PINHOLE"
    pinhole = "PINHOLE"
    simple_radial = "SIMPLE_RADIAL"
    radial = "RADIAL"
    opencv = "OPENCV"
    opencv_fisheye = "OPENCV_FISHEYE"
    full_opencv = "FULL_OPENCV"


class LinkMode(str, Enum):
    copy = "copy"
    symlink = "symlink"
    hardlink = "hardlink"


def echo_step(message: str) -> None:
    typer.secho(f"==> {message}", fg=typer.colors.CYAN, bold=True)


ArgValue = str | int | float | Path


def run(colmap: str, command: str, args: dict[str, ArgValue]) -> None:
    argv = [colmap, command]
    for key, value in args.items():
        argv += [f"--{key}", str(value)]
    typer.secho("    " + " ".join(argv), fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(argv)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def supported_options(colmap: str, command: str) -> set[str]:
    """Option names a given colmap subcommand accepts, from its own --help output."""
    result = subprocess.run(
        [colmap, command, "-h"], capture_output=True, text=True, check=False
    )
    return {
        token.lstrip("-").split()[0]
        for token in (result.stdout + result.stderr).split("\n")
        for token in [token.strip()]
        if token.startswith("--")
    }


def prefixed(options: set[str], candidates: list[str], suffix: str) -> str | None:
    """Pick whichever option-group prefix this colmap build actually uses.

    COLMAP <=3.x names these SiftExtraction/SiftMatching; 4.x renamed them to
    FeatureExtraction/FeatureMatching. Passing the wrong one is a hard error.
    """
    for prefix in candidates:
        name = f"{prefix}.{suffix}"
        if name in options:
            return name
    return None


def collect_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def place_images(files: list[Path], dest: Path, mode: LinkMode) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = dest / src.name
        if target.exists() or target.is_symlink():
            continue
        if mode is LinkMode.copy:
            shutil.copy2(src, target)
        elif mode is LinkMode.symlink:
            target.symlink_to(src.resolve())
        else:
            try:
                target.hardlink_to(src)
            except OSError:  # cross-device or unsupported fs
                shutil.copy2(src, target)


def ffprobe_for(ffmpeg: str) -> str:
    """The ffprobe that ships alongside the given ffmpeg, falling back to $PATH."""
    sibling = Path(ffmpeg).with_name("ffprobe") if "/" in ffmpeg else Path("ffprobe")
    return str(sibling) if shutil.which(str(sibling)) else "ffprobe"


def parse_timestamp(value: str) -> Optional[float]:
    """Seconds from an ffmpeg timestamp: 90, 1:30, 00:01:30.5. None if unparsable."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        return None
    seconds = 0.0
    for part in parts:
        try:
            seconds = seconds * 60 + float(part)
        except ValueError:
            return None
    return seconds


def probe_video(ffprobe: str, video: Path) -> tuple[Optional[float], Optional[float]]:
    """(duration in seconds, native frame rate) of the first video stream, or Nones."""
    argv = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError:
        return None, None
    if result.returncode != 0:
        return None, None

    duration = rate = None
    for line in result.stdout.split():
        if "/" in line:  # r_frame_rate comes as a rational, e.g. 30000/1001
            num, _, den = line.partition("/")
            try:
                rate = float(num) / float(den) if float(den) else None
            except ValueError:
                pass
        else:
            try:
                duration = float(line)
            except ValueError:
                pass
    return duration, rate


def auto_fps(seconds: float, native: Optional[float]) -> float:
    """Sampling rate that yields roughly TARGET_FRAMES over a clip of this length.

    Short clips keep every frame — a few seconds of footage has no frames to spare.
    Long ones are floored at MIN_AUTO_FPS instead of hitting the target exactly,
    since sparser sampling breaks feature tracking; cap those with --max-frames.
    """
    if seconds <= 0:
        return FALLBACK_FPS
    rate = max(TARGET_FRAMES / seconds, MIN_AUTO_FPS)
    if native and rate >= native:
        return 0.0  # every frame
    return round(rate, 2)


def resolve_fps(
    fps: float,
    ffmpeg: str,
    videos: list[Path],
    start: Optional[str],
    duration: Optional[str],
) -> float:
    """Turn --fps auto into a concrete rate, reporting what was chosen and why.

    Several videos share one rate, chosen from their combined length, so the
    whole capture lands near TARGET_FRAMES rather than each clip.
    """
    if fps != AUTO_FPS:
        return fps

    span = 0.0
    natives: list[float] = []
    for video in videos:
        total, native = probe_video(ffprobe_for(ffmpeg), video)
        if total is None:
            typer.secho(
                f"    could not probe {video.name} duration — falling back to {FALLBACK_FPS} fps",
                fg=typer.colors.YELLOW,
            )
            return FALLBACK_FPS
        # Sample against the span actually being extracted, not the whole file.
        clip = total
        if start is not None and (offset := parse_timestamp(start)) is not None:
            clip -= offset
        if duration is not None and (window := parse_timestamp(duration)) is not None:
            clip = min(clip, window)
        span += max(clip, 0.0)
        if native:
            natives.append(native)

    chosen = auto_fps(span, min(natives) if natives else None)
    typer.secho(
        f"    {span:.1f}s of video → "
        + ("every frame" if chosen <= 0 else f"{chosen} fps")
        + f" (~{TARGET_FRAMES} frames target)",
        fg=typer.colors.BRIGHT_BLACK,
    )
    return chosen


def extract_frames(
    ffmpeg: str,
    video: Path,
    dest: Path,
    fps: float,
    start: Optional[str],
    duration: Optional[str],
    quality: int,
    prefix: str = "",
) -> list[Path]:
    """Sample frames out of a video into dest/<prefix>frame_%06d.jpg."""
    dest.mkdir(parents=True, exist_ok=True)
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start is not None:  # before -i so ffmpeg seeks instead of decoding the head
        argv += ["-ss", start]
    if duration is not None:
        argv += ["-t", duration]
    argv += ["-i", str(video)]
    if fps > 0:
        argv += ["-vf", f"fps={fps}"]
    argv += ["-qscale:v", str(quality), "-vsync", "0", str(dest / f"{prefix}frame_%06d.jpg")]

    typer.secho("    " + " ".join(argv), fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(argv)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    return sorted(dest.glob(f"{prefix}frame_*.jpg"))


def thin_frames(frames: list[Path], limit: int) -> list[Path]:
    """Keep at most `limit` evenly spaced frames, deleting the rest."""
    if limit <= 0 or len(frames) <= limit:
        return frames
    step = len(frames) / limit
    keep = {frames[int(i * step)] for i in range(limit)}
    for frame in frames:
        if frame not in keep:
            frame.unlink()
    return sorted(keep)


def pick_largest_model(sparse_root: Path) -> Path:
    """COLMAP writes sparse/0, sparse/1, ... — return the one with most images."""
    models = [d for d in sorted(sparse_root.iterdir()) if (d / "images.bin").exists()]
    if not models:
        typer.secho(
            "Reconstruction produced no model — check feature matching settings.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return max(models, key=lambda d: (d / "images.bin").stat().st_size)


@app.command()
def main(
    sources: list[Path] = typer.Argument(
        ...,
        exists=True,
        readable=True,
        show_default=False,
        help="Folder with the input images (e.g. .../images_long1600), or one or more "
        "video files to sample frames from. Frames from several videos go into one "
        "dataset, reconstructed together.",
    ),
    output: Path = typer.Argument(
        ..., help="Destination dataset root to create (person-hall style)."
    ),
    matcher: Optional[Matcher] = typer.Option(
        None,
        "--matcher",
        "-m",
        help="Matching strategy. Defaults to 'exhaustive' for image folders and several "
        "videos, and 'sequential' for a single video; 'vocab_tree' scales to thousands "
        "of unordered images.",
    ),
    camera_model: CameraModel = typer.Option(
        CameraModel.opencv, "--camera-model", "-c", help="COLMAP camera model."
    ),
    single_camera: bool = typer.Option(
        True,
        "--single-camera/--per-image-camera",
        help="Share one intrinsics block across all images (correct for one physical camera).",
    ),
    link_mode: LinkMode = typer.Option(
        LinkMode.copy,
        "--link-mode",
        "-l",
        help="How to populate images/: copy (portable), symlink or hardlink (saves disk). "
        "Ignored for video sources — frames are always written out.",
    ),
    fps: float = typer.Option(
        0.0,
        "--fps",
        help=f"Video only: frames sampled per second. Default 0 keeps every frame — cap "
        f"the count with --max-frames. Pass {AUTO_FPS:g} to adapt the rate to the clip's "
        f"length instead, aiming for ~{TARGET_FRAMES} frames (never below {MIN_AUTO_FPS} fps).",
    ),
    max_frames: int = typer.Option(
        0,
        "--max-frames",
        help="Video only: cap the number of frames, dropping evenly spaced extras. With "
        "several videos the cap is for all of them, shared in proportion. 0 = no cap.",
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Video only: skip to this timestamp (ffmpeg -ss, e.g. 00:00:10), in every video.",
    ),
    duration: Optional[str] = typer.Option(
        None,
        "--duration",
        help="Video only: how much to read from --start (ffmpeg -t), in every video.",
    ),
    frame_quality: int = typer.Option(
        2,
        "--frame-quality",
        min=1,
        max=31,
        help="Video only: JPEG quality of extracted frames (ffmpeg -qscale:v, 1 = best).",
    ),
    ffmpeg_bin: str = typer.Option("ffmpeg", "--ffmpeg", help="Path to the ffmpeg executable."),
    vocab_tree_path: Optional[Path] = typer.Option(
        None,
        "--vocab-tree",
        exists=True,
        dir_okay=False,
        help="Vocabulary tree file, required for --matcher vocab_tree.",
    ),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use CUDA for SIFT and matching."),
    max_image_size: int = typer.Option(
        3200, "--max-image-size", help="Downscale limit for feature extraction."
    ),
    keep_binary: bool = typer.Option(
        False,
        "--keep-binary/--txt-only",
        help="Also keep the .bin model next to the exported .txt files.",
    ),
    keep_rig_files: bool = typer.Option(
        False,
        "--keep-rig-files/--no-rig-files",
        help="Keep frames.txt/rigs.txt that COLMAP 4.x emits. Off by default so "
        "sparse/ holds only cameras/images/points3D, like the person-hall layout.",
    ),
    colmap_bin: str = typer.Option("colmap", "--colmap", help="Path to the colmap executable."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite an existing non-empty output directory."
    ),
) -> None:
    """Run the full COLMAP pipeline and lay the result out as a person-hall style dataset."""
    videos = [s for s in sources if s.is_file()]
    folders = [s for s in sources if not s.is_file()]
    for video in videos:
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            typer.secho(
                f"{video} is a file but not a recognised video "
                f"({', '.join(sorted(VIDEO_SUFFIXES))}). Pass a folder of images instead.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
    if folders and (videos or len(folders) > 1):
        typer.secho(
            "Pass either one folder of images, or one or more videos — not a mix.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    is_video = bool(videos)
    source = folders[0] if folders else videos[0]

    if matcher is None:
        # One clip's frames are in order; several clips have to be matched
        # against each other too, which only exhaustive matching does.
        matcher = Matcher.sequential if len(videos) == 1 else Matcher.exhaustive
        if is_video:
            typer.secho(
                "Video source — using --matcher sequential (frames are ordered)."
                if len(videos) == 1
                else f"{len(videos)} videos — using --matcher exhaustive (clips must match each other).",
                fg=typer.colors.YELLOW,
            )

    if is_video and shutil.which(ffmpeg_bin) is None:
        typer.secho(f"ffmpeg executable not found: {ffmpeg_bin}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    if matcher is Matcher.vocab_tree and vocab_tree_path is None:
        typer.secho(
            "--matcher vocab_tree requires --vocab-tree /path/to/vocab_tree.bin",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if shutil.which(colmap_bin) is None:
        typer.secho(f"colmap executable not found: {colmap_bin}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    images_in: list[Path] = []
    if not is_video:
        images_in = collect_images(source)
        if not images_in:
            typer.secho(f"No images found in {source}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    if output.exists() and any(output.iterdir()):
        if not force:
            typer.secho(
                f"{output} already exists and is not empty — pass --force to overwrite.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.secho(f"Removing existing {output}", fg=typer.colors.YELLOW)
        shutil.rmtree(output)

    images_dir = output / "images"
    database = output / "database.db"
    sparse_dir = output / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if is_video:
        fps = resolve_fps(fps, ffmpeg_bin, videos, start, duration)
        frames: list[Path] = []
        for n, video in enumerate(videos, 1):
            echo_step(f"Extracting frames from {video.name} into {images_dir}")
            # A single video keeps plain frame_NNNNNN names; several get a
            # per-clip prefix so their frames neither collide nor interleave.
            prefix = "" if len(videos) == 1 else f"v{n:02d}_"
            got = extract_frames(
                ffmpeg_bin, video, images_dir, fps, start, duration, frame_quality, prefix
            )
            if not got:
                typer.secho(f"ffmpeg produced no frames from {video}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            frames += got
        # Sorted by name the clips follow one another, so evenly spaced
        # thinning takes from each in proportion to its length.
        images_in = thin_frames(frames, max_frames)
        typer.secho(
            f"    {len(images_in)} frames"
            + (f" (thinned from {len(frames)})" if len(images_in) < len(frames) else "")
            + (f" from {len(videos)} videos" if len(videos) > 1 else ""),
            fg=typer.colors.BRIGHT_BLACK,
        )
    else:
        echo_step(f"Staging {len(images_in)} images into {images_dir} ({link_mode.value})")
        place_images(images_in, images_dir, link_mode)

    echo_step("Extracting SIFT features")
    extract_opts = supported_options(colmap_bin, "feature_extractor")
    extract_args: dict[str, ArgValue] = {
        "database_path": database,
        "image_path": images_dir,
        "ImageReader.camera_model": camera_model.value,
        "ImageReader.single_camera": int(single_camera),
    }
    groups = ["FeatureExtraction", "SiftExtraction"]
    if name := prefixed(extract_opts, groups, "use_gpu"):
        extract_args[name] = int(gpu)
    if name := prefixed(extract_opts, groups, "max_image_size"):
        extract_args[name] = max_image_size
    run(colmap_bin, "feature_extractor", extract_args)

    echo_step(f"Matching features ({matcher.value})")
    match_command = f"{matcher.value}_matcher"
    match_opts = supported_options(colmap_bin, match_command)
    match_args: dict[str, ArgValue] = {"database_path": database}
    if name := prefixed(match_opts, ["FeatureMatching", "SiftMatching"], "use_gpu"):
        match_args[name] = int(gpu)
    if vocab_tree_path is not None:
        match_args["VocabTreeMatching.vocab_tree_path"] = vocab_tree_path
    run(colmap_bin, match_command, match_args)

    echo_step("Running incremental mapper")
    run(
        colmap_bin,
        "mapper",
        {
            "database_path": database,
            "image_path": images_dir,
            "output_path": sparse_dir,
        },
    )

    model_dir = pick_largest_model(sparse_dir)
    echo_step(f"Exporting model {model_dir.name} to TXT in {sparse_dir}")
    run(
        colmap_bin,
        "model_converter",
        {
            "input_path": model_dir,
            "output_path": sparse_dir,
            "output_type": "TXT",
        },
    )

    if not keep_binary:
        for sub in sorted(sparse_dir.iterdir()):
            if sub.is_dir():
                shutil.rmtree(sub)

    if not keep_rig_files:
        for extra in ("frames.txt", "rigs.txt"):
            (sparse_dir / extra).unlink(missing_ok=True)

    registered = sum(
        1
        for line in (sparse_dir / "images.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ) // 2

    echo_step("Done")
    typer.secho(
        f"{registered}/{len(images_in)} images registered → {output}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    if registered < len(images_in):
        typer.secho(
            "Some images were not registered. Try a different --matcher, "
            "raise --max-image-size, or inspect the model in `colmap gui`.",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":
    sys.exit(app())
