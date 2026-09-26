# lichtfeld-preparater

Converts a flat folder of images — or one or more video files — into a COLMAP dataset laid out
the way downstream training tools expect (the `person-hall` layout).

## Input

Any directory of images:

```
~/Datasets/belval/images_long1600/
  DJI_0072.JPG
  DJI_0073.JPG
  ...
```

…or a single video file (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.webm`, `.mpg`,
`.mpeg`, `.mts`, `.insv`), from which `ffmpeg` samples frames into `images/`.

## Output

```
<output>/
  images/         staged source images
  database.db     COLMAP feature/match database
  sparse/         reconstructed model, TXT format
    cameras.txt
    images.txt
    points3D.txt
```

## Usage

```bash
./lichtfeld_preparater.py ~/Datasets/belval/images_long1600 ~/Datasets/belval-colmap
```

From a video — frames are sampled into `images/` as `frame_000001.jpg`, …:

```bash
./lichtfeld_preparater.py ~/Videos/hall.mp4 ~/Datasets/hall-colmap
```

Several videos of the same scene — walked in separate takes, say — go into one
dataset and are reconstructed together. Their frames are named after the clip
(`v01_frame_000001.jpg`, `v02_frame_000001.jpg`, …), and matching defaults to
`exhaustive`, since frames of one clip must be matched against the others:

```bash
./lichtfeld_preparater.py hall-take1.mp4 hall-take2.mp4 ~/Datasets/hall-colmap --fps -1 --max-frames 300
```

`--fps -1` then picks one rate from the clips' combined length, `--max-frames`
caps all of them together (each keeps its share, by length), and `--start` /
`--duration` apply to every clip.

Every frame of the clip is kept by default. Long captures can therefore produce a
lot of images — cap them with `--max-frames`, or thin the sampling with `--fps`.
Passing `--fps -1` restores the adaptive rate, which targets ~200 frames from the
clip's duration and never drops below 1 fps.

Sample a slice of a long clip and cap the frame count:

```bash
./lichtfeld_preparater.py hall.mp4 OUT --start 00:01:30 --duration 45 --fps 4 --max-frames 300
```

Drone or video captures shot in sequence match much faster with `sequential`
(the default for video sources):

```bash
./lichtfeld_preparater.py IN OUT --matcher sequential
```

Save disk space instead of duplicating 3 GB of JPEGs:

```bash
./lichtfeld_preparater.py IN OUT --link-mode hardlink
```

Large unordered collections need a vocabulary tree
([download](https://demuc.de/colmap/#download)):

```bash
./lichtfeld_preparater.py IN OUT --matcher vocab_tree --vocab-tree ~/vocab_tree_flickr100K_words256K.bin
```

### Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--matcher` / `-m` | `exhaustive` (images, several videos), `sequential` (one video) | `exhaustive`, `sequential`, `vocab_tree`, `spatial` |
| `--camera-model` / `-c` | `OPENCV` | COLMAP camera model |
| `--single-camera` / `--per-image-camera` | single | Share one intrinsics block across all images |
| `--link-mode` / `-l` | `copy` | `copy`, `symlink`, `hardlink` (image folders only) |
| `--fps` | `0` (every frame) | Video: frames sampled per second; `-1` adapts the rate to the clip's duration, targeting ~200 frames |
| `--max-frames` | `0` | Video: cap frame count, dropping evenly spaced extras |
| `--start` | — | Video: seek to this timestamp before sampling (`ffmpeg -ss`) |
| `--duration` | — | Video: how much to read from `--start` (`ffmpeg -t`) |
| `--frame-quality` | `2` | Video: JPEG quality of extracted frames (1 = best) |
| `--ffmpeg` | `ffmpeg` | Path to the ffmpeg executable |
| `--gpu` / `--no-gpu` | GPU | CUDA for SIFT extraction and matching |
| `--max-image-size` | `3200` | Downscale limit during feature extraction |
| `--keep-binary` / `--txt-only` | txt-only | Keep the `.bin` model alongside the `.txt` export |
| `--keep-rig-files` / `--no-rig-files` | dropped | Keep `frames.txt` / `rigs.txt` emitted by COLMAP 4.x |
| `--colmap` | `colmap` | Path to the COLMAP executable |
| `--force` / `-f` | off | Overwrite a non-empty output directory |

Run `./lichtfeld_preparater.py --help` for the full list.

## Requirements

- `colmap` on `PATH` (CUDA build recommended; use `--no-gpu` otherwise)
- `ffmpeg` on `PATH` — only needed for video sources
- Python 3.10+ and `typer`

```bash
pip install -r requirements.txt
```

## Nix

The flake installs the script as `lichtfeld-preparater`, with `colmap` and
`ffmpeg` wired into its `PATH`:

```bash
nix run github:paulusx/lichtfeld-preparater -- ~/Videos/hall.mp4 ~/Datasets/hall-colmap
nix profile install github:paulusx/lichtfeld-preparater
```

Both are appended to `PATH`, not prepended, so a CUDA `colmap` you already have
still takes precedence — as do `--colmap` and `--ffmpeg`.

In a system or Home Manager flake:

```nix
inputs.lichtfeld-preparater.url = "github:paulusx/lichtfeld-preparater";

# then, in your package list:
inputs.lichtfeld-preparater.packages.${pkgs.system}.default
```

On a channel-based (non-flake) NixOS, `package.nix` is `callPackage`-shaped, so an
overlay can import it and substitute its own COLMAP:

```nix
lichtfeld-preparater = prev.callPackage "${src}/package.nix" {
  colmap = final.colmapWithCuda;
};
```

`nix develop` gives a shell with Python, `typer`, `colmap` and `ffmpeg` for
running `./lichtfeld_preparater.py` straight from the checkout.

## Notes

- Option names differ between COLMAP releases (`SiftExtraction.*` in 3.x became
  `FeatureExtraction.*` in 4.x). The script reads each subcommand's `--help` and
  picks whichever your build accepts, so it works on both.
- COLMAP 4.x also writes `frames.txt` and `rigs.txt`. These are removed by default
  so `sparse/` matches the `person-hall` layout; keep them with `--keep-rig-files`.
- Video frames are written straight into `images/`, so `--link-mode` does not apply.
  Keeping every frame gives the mapper the most overlap to work with, but matching
  cost grows with the frame count — lower `--fps` or set `--max-frames` on long or
  slow-moving captures to keep reconstruction time bounded.
- The mapper can split a capture into several models (`sparse/0`, `sparse/1`, …).
  The script exports the largest one and drops the rest unless `--keep-binary` is set.
- The final line reports how many images were registered. If that count is well
  below the input count, try another `--matcher`, raise `--max-image-size`, or
  inspect the result in `colmap gui`.
- `output/` in an existing `person-hall` dataset holds training checkpoints, not
  conversion results, so this script does not create it.
