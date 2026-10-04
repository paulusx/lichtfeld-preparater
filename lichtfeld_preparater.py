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
    ./lichtfeld_preparater.py ~/Videos/hall.mp4 ~/Datasets/hall-colmap --every 5
    ./lichtfeld_preparater.py ~/Videos/hall-1.mp4 ~/Videos/hall-2.mp4 ~/Datasets/hall-colmap
    ./lichtfeld_preparater.py ~/Videos/VID_..._00_005.insv ~/Datasets/park-colmap --fps -1

360° videos (Insta360 .insv dual fisheye, or an equirectangular export) are cut
into several flat views per frame, tied together as a COLMAP camera rig.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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


class Panorama(str, Enum):
    auto = "auto"
    off = "off"
    equirect = "equirect"
    dual_fisheye = "dual-fisheye"


class FrameFormat(str, Enum):
    jpg = "jpg"
    png = "png"


def encode_args(fmt: FrameFormat, quality: int) -> list[str]:
    """ffmpeg output options for one extracted frame; PNG is lossless, so no quality."""
    return ["-qscale:v", str(quality)] if fmt is FrameFormat.jpg else []


# How a 360° frame is cut into flat views. Each view is a square pinhole camera
# of VIEW_FOV degrees; neighbours overlap so features carry across them.
VIEW_FOV = 90.0
EQUIRECT_YAWS = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
# Dual fisheye: views stay within one lens each, so none straddles the stitch
# seam. Yaws are relative to that lens's axis; the back lens faces yaw 180.
LENS_YAWS = (-45.0, 0.0, 45.0)
DEFAULT_LENS_FOV = 200.0  # Insta360 X-series lenses see a bit over a hemisphere


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


def sampling_filter(fps: float, every: int) -> Optional[str]:
    """The ffmpeg filter that picks which frames to keep, or None for all of them."""
    if every > 1:
        # Counted from the first decoded frame, i.e. from --start.
        return f"select=not(mod(n\\,{every}))"
    if fps > 0:
        return f"fps={fps}"
    return None


def extract_frames(
    ffmpeg: str,
    video: Path,
    dest: Path,
    sample: Optional[str],
    start: Optional[str],
    duration: Optional[str],
    quality: int,
    fmt: FrameFormat,
    prefix: str = "",
) -> list[Path]:
    """Sample frames out of a video into dest/<prefix>frame_%06d.<fmt>."""
    dest.mkdir(parents=True, exist_ok=True)
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start is not None:  # before -i so ffmpeg seeks instead of decoding the head
        argv += ["-ss", start]
    if duration is not None:
        argv += ["-t", duration]
    argv += ["-i", str(video)]
    if sample is not None:
        argv += ["-vf", sample]
    argv += [*encode_args(fmt, quality), "-vsync", "0", str(dest / f"{prefix}frame_%06d.{fmt.value}")]

    typer.secho("    " + " ".join(argv), fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(argv)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    return sorted(dest.glob(f"{prefix}frame_*.{fmt.value}"))


@dataclass(frozen=True)
class View:
    """One flat view cut out of a 360° video."""

    source: str  # ffmpeg filter input: a stream label, then a crop for side-by-side lenses
    yaw: float  # in the source: equirect yaw, or offset from the lens axis
    rig_yaw: float  # direction of the view within the rig, which faces yaw 0
    fisheye: bool
    size: int  # square output, in pixels


def video_streams(ffprobe: str, video: Path) -> list[tuple[str, int, int]]:
    """(ffmpeg label, width, height) of each real video stream — cover art is skipped."""
    argv = [
        ffprobe, "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=width,height:stream_disposition=attached_pic",
        "-of", "csv=p=0",
        str(video),
    ]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError:
        return []
    streams = []
    for n, line in enumerate(result.stdout.splitlines()):
        fields = line.strip().split(",")
        if len(fields) < 3 or fields[2] == "1":
            continue
        try:
            streams.append((f"[0:v:{n}]", int(fields[0]), int(fields[1])))
        except ValueError:
            pass
    return streams


def even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def panorama_views(
    ffprobe: str, video: Path, mode: Panorama, lens_fov: float
) -> list[View]:
    """The flat views to cut out of this video, or [] if it is an ordinary one.

    auto recognises an .insv with one stream per lens (Insta360 X-series), an
    .insv with both lenses side by side in one stream (older models), and any
    other 2:1 video as equirectangular.
    """
    if mode is Panorama.off:
        return []
    streams = video_streams(ffprobe, video)
    if not streams:
        return []
    first, width, height = streams[0]
    two_lenses = len(streams) >= 2 and streams[1][1:] == (width, height) and width == height
    side_by_side = len(streams) == 1 and width == 2 * height

    if mode is Panorama.auto:
        if two_lenses or (side_by_side and video.suffix.lower() == ".insv"):
            mode = Panorama.dual_fisheye
        elif side_by_side:
            mode = Panorama.equirect
        else:
            return []

    if mode is Panorama.equirect:
        size = even(width * VIEW_FOV / 360)
        return [View(first, yaw, yaw, False, size) for yaw in EQUIRECT_YAWS]

    if two_lenses:
        lenses = [first, streams[1][0]]
    elif side_by_side:
        lenses = [f"{first}crop=iw/2:ih:0:0,", f"{first}crop=iw/2:ih:iw/2:0,"]
    else:
        typer.secho(
            f"{video.name}: expected two square lens streams, or both lenses side by "
            f"side in one 2:1 stream, got {[s[1:] for s in streams]}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    size = even(height * VIEW_FOV / lens_fov)
    return [
        View(source, yaw, (axis + yaw) % 360, True, size)
        for source, axis in zip(lenses, (0.0, 180.0))
        for yaw in LENS_YAWS
    ]


def view_filter(view: View, lens_fov: float) -> str:
    projection = (
        f"input=fisheye:ih_fov={lens_fov:g}:iv_fov={lens_fov:g}"
        if view.fisheye
        else "input=e"
    )
    yaw = (view.yaw + 180) % 360 - 180  # v360 only takes yaw in [-180, 180]
    return (
        f"v360={projection}:output=flat:h_fov={VIEW_FOV:g}:v_fov={VIEW_FOV:g}"
        f":yaw={yaw:g}:w={view.size}:h={view.size}"
    )


def view_prefix(prefix: str, n: int) -> str:
    return f"{prefix}c{n}_"


def extract_views(
    ffmpeg: str,
    video: Path,
    views: list[View],
    lens_fov: float,
    dest: Path,
    sample: Optional[str],
    start: Optional[str],
    duration: Optional[str],
    quality: int,
    fmt: FrameFormat,
    prefix: str = "",
) -> list[list[Path]]:
    """Sample a 360° video into dest/<prefix>c<N>_frame_%06d.<fmt>, one file per view.

    Returns the frames as groups: every view of one moment. The video is decoded
    once; each lens is resampled first, so all views share the same timestamps.
    """
    dest.mkdir(parents=True, exist_ok=True)
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start is not None:
        argv += ["-ss", start]
    if duration is not None:
        argv += ["-t", duration]
    argv += ["-i", str(video)]

    graph = []
    sources = list(dict.fromkeys(view.source for view in views))
    for s, source in enumerate(sources):
        mine = [n for n, view in enumerate(views) if view.source == source]
        chain = source + (f"{sample}," if sample is not None else "")
        graph.append(f"{chain}split={len(mine)}" + "".join(f"[s{s}_{n}]" for n in mine))
        for n in mine:
            graph.append(f"[s{s}_{n}]{view_filter(views[n], lens_fov)}[o{n}]")
    argv += ["-filter_complex", ";".join(graph)]
    for n in range(len(views)):
        argv += [
            "-map", f"[o{n}]",
            *encode_args(fmt, quality),
            "-fps_mode", "passthrough",
            str(dest / f"{view_prefix(prefix, n)}frame_%06d.{fmt.value}"),
        ]

    typer.secho("    " + " ".join(argv), fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(argv)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)

    # A moment only counts if every view of it came out; drop stragglers.
    per_view = [
        {f.name[len(view_prefix(prefix, n)):]: f for f in dest.glob(f"{view_prefix(prefix, n)}frame_*.{fmt.value}")}
        for n in range(len(views))
    ]
    common = set(per_view[0]).intersection(*per_view[1:])
    for files in per_view:
        for name, path in files.items():
            if name not in common:
                path.unlink()
    return [[files[name] for files in per_view] for name in sorted(common)]


def thin_frames(frames: list[list[Path]], limit: int) -> list[list[Path]]:
    """Keep at most `limit` evenly spaced moments (all views of each), deleting the rest."""
    if limit <= 0 or len(frames) <= limit:
        return frames
    step = len(frames) / limit
    keep = {int(i * step) for i in range(limit)}
    for n, group in enumerate(frames):
        if n not in keep:
            for frame in group:
                frame.unlink()
    return [frames[n] for n in sorted(keep)]


def cam_from_rig(yaw: float) -> list[float]:
    """COLMAP quaternion (w, x, y, z) of a camera turned `yaw` degrees to the right.

    COLMAP cameras look down +z with y pointing down, so a yaw is a rotation
    about y; cam_from_rig is its inverse.
    """
    half = math.radians(-yaw) / 2
    return [math.cos(half), 0.0, math.sin(half), 0.0]


def pinhole_params(size: int) -> str:
    focal = size / 2 / math.tan(math.radians(VIEW_FOV) / 2)
    return f"{focal:.4f},{focal:.4f},{size / 2:g},{size / 2:g}"


@dataclass
class ImageGroup:
    """Images that go through feature extraction with the same camera settings."""

    names: list[str]
    camera_model: str
    single_camera: bool
    camera_params: Optional[str] = None


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
    every: int = typer.Option(
        1,
        "--every",
        min=1,
        help="Video only: keep one frame in every N, counted in the video's own frames. "
        "Instead of --fps; 1 = every frame. Combine with --max-frames 0 to keep them all.",
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
    frame_format: FrameFormat = typer.Option(
        FrameFormat.jpg,
        "--frame-format",
        help="Video only: file format of extracted frames. png is lossless but several "
        "times larger; --frame-quality then has no effect.",
    ),
    panorama: Panorama = typer.Option(
        Panorama.auto,
        "--panorama",
        help="Video only: how to treat 360° video. auto detects an Insta360 .insv (dual "
        "fisheye) or a 2:1 equirectangular video per clip; the others force one kind on "
        f"every clip. Each 360° frame becomes {len(EQUIRECT_YAWS)} flat {VIEW_FOV:g}° views.",
    ),
    lens_fov: float = typer.Option(
        DEFAULT_LENS_FOV,
        "--lens-fov",
        help="Dual fisheye only: field of view of each lens, in degrees.",
    ),
    rig: bool = typer.Option(
        True,
        "--rig/--no-rig",
        help="Tie the views of each 360° frame together as a COLMAP rig, so they are "
        "posed as one. --no-rig reconstructs them as independent images.",
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
    if every > 1 and fps != 0:
        typer.secho("Pass either --fps or --every, not both.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
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

    layouts = {
        video: panorama_views(ffprobe_for(ffmpeg_bin), video, panorama, lens_fov)
        for video in videos
    }
    for video, views in layouts.items():
        if views:
            kind = "dual fisheye" if views[0].fisheye else "equirectangular"
            typer.secho(
                f"{video.name}: 360° {kind} — {len(views)} views of {views[0].size}px per frame",
                fg=typer.colors.YELLOW,
            )

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

    # Flat images share the user's camera settings; each view of a 360° video
    # is a camera of its own, with intrinsics known from the projection.
    flat = ImageGroup([], camera_model.value, single_camera)
    groups = [flat]
    rigs: list[dict] = []
    if is_video:
        fps = resolve_fps(fps, ffmpeg_bin, videos, start, duration)
        sample = sampling_filter(fps, every)
        frames: list[list[Path]] = []
        clips: list[tuple[Path, str, list[list[Path]]]] = []
        for n, video in enumerate(videos, 1):
            echo_step(f"Extracting frames from {video.name} into {images_dir}")
            # A single video keeps plain frame_NNNNNN names; several get a
            # per-clip prefix so their frames neither collide nor interleave.
            prefix = "" if len(videos) == 1 else f"v{n:02d}_"
            if views := layouts[video]:
                got = extract_views(
                    ffmpeg_bin, video, views, lens_fov, images_dir,
                    sample, start, duration, frame_quality, frame_format, prefix,
                )
            else:
                got = [
                    [frame]
                    for frame in extract_frames(
                        ffmpeg_bin, video, images_dir, sample, start, duration, frame_quality, frame_format,
                        prefix,
                    )
                ]
            if not got:
                typer.secho(f"ffmpeg produced no frames from {video}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            clips.append((video, prefix, got))
            frames += got
        # Sorted by name the clips follow one another, so evenly spaced
        # thinning takes from each in proportion to its length.
        kept = thin_frames(frames, max_frames)
        images_in = [image for group in kept for image in group]
        typer.secho(
            f"    {len(kept)} frames"
            + (f" (thinned from {len(frames)})" if len(kept) < len(frames) else "")
            + (f" from {len(videos)} videos" if len(videos) > 1 else "")
            + (f", {len(images_in)} images" if len(images_in) > len(kept) else ""),
            fg=typer.colors.BRIGHT_BLACK,
        )

        present = {image.name for image in images_in}
        for video, prefix, got in clips:
            views = layouts[video]
            if not views:
                flat.names += [g[0].name for g in got if g[0].name in present]
                continue
            cameras = []
            for n, view in enumerate(views):
                groups.append(
                    ImageGroup(
                        [g[n].name for g in got if g[n].name in present],
                        "PINHOLE",
                        True,
                        pinhole_params(view.size),
                    )
                )
                camera: dict = {"image_prefix": view_prefix(prefix, n)}
                if view.rig_yaw == 0:
                    camera["ref_sensor"] = True
                else:
                    camera["cam_from_rig_rotation"] = cam_from_rig(view.rig_yaw)
                    camera["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
                cameras.append(camera)
            # COLMAP adds sensors in order and wants the reference one first.
            cameras.sort(key=lambda camera: not camera.get("ref_sensor", False))
            rigs.append({"cameras": cameras})
    else:
        echo_step(f"Staging {len(images_in)} images into {images_dir} ({link_mode.value})")
        place_images(images_in, images_dir, link_mode)
        flat.names = [image.name for image in images_in]

    echo_step("Extracting SIFT features")
    extract_opts = supported_options(colmap_bin, "feature_extractor")
    feature_args: dict[str, ArgValue] = {}
    option_groups = ["FeatureExtraction", "SiftExtraction"]
    if name := prefixed(extract_opts, option_groups, "use_gpu"):
        feature_args[name] = int(gpu)
    if name := prefixed(extract_opts, option_groups, "max_image_size"):
        feature_args[name] = max_image_size
    with tempfile.TemporaryDirectory() as scratch:
        for n, group in enumerate(g for g in groups if g.names):
            image_list = Path(scratch) / f"images_{n}.txt"
            image_list.write_text("".join(f"{name}\n" for name in group.names))
            extract_args: dict[str, ArgValue] = {
                "database_path": database,
                "image_path": images_dir,
                "image_list_path": image_list,
                "ImageReader.camera_model": group.camera_model,
                "ImageReader.single_camera": int(group.single_camera),
            }
            if group.camera_params is not None:
                extract_args["ImageReader.camera_params"] = group.camera_params
            run(colmap_bin, "feature_extractor", extract_args | feature_args)

        if rigs and rig:
            echo_step(f"Configuring {len(rigs)} camera rig(s) for the 360° views")
            rig_config = Path(scratch) / "rigs.json"
            rig_config.write_text(json.dumps(rigs, indent=2))
            run(
                colmap_bin,
                "rig_configurator",
                {"database_path": database, "rig_config_path": rig_config},
            )

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
