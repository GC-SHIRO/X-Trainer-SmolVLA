#!/usr/bin/env python
"""Create a camera-orientation-corrected copy of an X-trainer v2.1 dataset.

The source must be the video-backed LeRobot v2.1 layout emitted by
``scripts/xtrainer/convert_raw_to_lerobot_2_1.py``. The source is never
modified. The output dataset applies the camera transforms needed to match the
deployment input:

* ``observation.images.top``: unchanged;
* ``observation.images.left_wrist``: unchanged;
* ``observation.images.right_wrist``: vertical then horizontal flip (180 deg).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np


CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
VIDEO_FILTERS = {
    "observation.images.right_wrist": "vflip,hflip",
}


def transform_frame(camera_key: str, image: np.ndarray) -> np.ndarray:
    """Apply this converter's orientation rule to one RGB HWC image."""
    if camera_key == "observation.images.top":
        return np.ascontiguousarray(image)
    if camera_key == "observation.images.left_wrist":
        return np.ascontiguousarray(image)
    if camera_key == "observation.images.right_wrist":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise KeyError(f"Unsupported camera key: {camera_key}")


def _video_paths(root: Path, camera_key: str) -> dict[Path, Path]:
    videos_root = root / "videos"
    paths = sorted(videos_root.glob(f"chunk-*/{camera_key}/episode_*.mp4"))
    return {
        path.relative_to(videos_root).parent.parent / path.name: path
        for path in paths
    }


def _validate_source(root: Path) -> dict[str, dict[Path, Path]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)

    features = info.get("features", {})
    invalid = [key for key in CAMERA_KEYS if features.get(key, {}).get("dtype") != "video"]
    if invalid:
        raise ValueError(
            "This tool supports only video-backed X-trainer v2.1 datasets; "
            f"these camera features are missing or not videos: {', '.join(invalid)}"
        )

    videos = {key: _video_paths(root, key) for key in CAMERA_KEYS}
    expected = set(videos[CAMERA_KEYS[0]])
    if not expected:
        raise RuntimeError("No top-camera MP4 files found under videos/chunk-*/.")
    for key, paths in videos.items():
        found = set(paths)
        if found != expected:
            missing = sorted(str(path) for path in expected - found)
            extra = sorted(str(path) for path in found - expected)
            raise ValueError(f"Video files for {key} do not match the top camera (missing={missing}, extra={extra})")
    return videos


def _assert_safe_output(source: Path, output: Path) -> None:
    if source == output or output in source.parents or source in output.parents:
        raise ValueError("--output-root must be a separate directory, not the source or one of its ancestors/children")


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    if not output.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output path already exists: {output}. Pass --overwrite-output to replace it.")
    if output.is_dir():
        shutil.rmtree(output)
    else:
        output.unlink()


def _run_ffmpeg(source: Path, destination: Path, video_filter: str, crf: int, preset: str) -> None:
    temporary = destination.with_name(f".{destination.stem}.transforming.mp4")
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-an",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed ({completed.returncode}) while transforming {source}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg created no video output for {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def transform_dataset(
    source_root: Path,
    output_root: Path,
    *,
    overwrite_output: bool = False,
    crf: int = 18,
    preset: str = "medium",
    dry_run: bool = False,
) -> None:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Input dataset directory does not exist: {source_root}")
    _assert_safe_output(source_root, output_root)
    videos = _validate_source(source_root)

    episode_count = len(videos[CAMERA_KEYS[0]])
    if dry_run:
        print(f"Validated {episode_count} episodes in {source_root}")
        print("Would copy the dataset, preserve top and left videos, and rotate right videos 180 degrees.")
        return

    _prepare_output(output_root, overwrite=overwrite_output)
    shutil.copytree(source_root, output_root)
    for camera_key, video_filter in VIDEO_FILTERS.items():
        for source_video in videos[camera_key].values():
            destination = output_root / "videos" / source_video.relative_to(source_root / "videos")
            _run_ffmpeg(destination, destination, video_filter, crf, preset)

    print(f"Created transformed dataset: {output_root}")
    print(f"Episodes: {episode_count}; top=unchanged, left=unchanged, right=vflip+hflip")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform X-trainer v2.1 dataset camera orientations into a new dataset.")
    parser.add_argument("--input-root", required=True, help="Existing video-backed LeRobot v2.1 dataset directory")
    parser.add_argument("--output-root", required=True, help="New output dataset directory; source is never modified")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality setting for transformed videos (default: 18)")
    parser.add_argument("--preset", default="medium", help="FFmpeg libx264 preset (default: medium)")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print planned work without creating output")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    transform_dataset(
        Path(args.input_root),
        Path(args.output_root),
        overwrite_output=args.overwrite_output,
        crf=args.crf,
        preset=args.preset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
