#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
REQUIRED_DATA_KEYS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
)
VECTOR_KEYS = ("observation.state", "action")
EXPECTED_VERSION = "v2.1"
EXPECTED_DIM = 14


@dataclass
class ValidationReport:
    root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_episodes: int = 0
    total_frames: int = 0
    sampled_episodes: int = 0
    sampled_frames: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _load_json(path: Path, report: ValidationReport) -> dict[str, Any]:
    if not path.exists():
        report.error(f"Missing required file: {path.relative_to(report.root)}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        report.error(f"Could not parse {path.relative_to(report.root)}: {exc}")
        return {}


def _load_jsonl(path: Path, report: ValidationReport) -> list[dict[str, Any]]:
    if not path.exists():
        report.error(f"Missing required file: {path.relative_to(report.root)}")
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as exc:
        report.error(f"Could not parse {path.relative_to(report.root)} line {line_no}: {exc}")
    return rows


def _feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...] | None:
    feature = (info.get("features") or {}).get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _is_video_feature(info: dict[str, Any], key: str) -> bool:
    feature = (info.get("features") or {}).get(key)
    return isinstance(feature, dict) and feature.get("dtype") == "video"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_data_file(root: Path, episode_index: int) -> Path:
    return root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _resolve_video_file(root: Path, camera_key: str, episode_index: int) -> Path:
    return root / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4"


def _validate_info(info: dict[str, Any], report: ValidationReport) -> None:
    version = info.get("codebase_version") or info.get("format_version")
    if version != EXPECTED_VERSION:
        report.error(f"Expected LeRobot dataset version {EXPECTED_VERSION}, got {version!r}")

    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        report.error(f"meta/info.json must declare a positive fps, got {fps!r}")

    for key in VECTOR_KEYS:
        shape = _feature_shape(info, key)
        if shape != (EXPECTED_DIM,):
            report.error(f"Feature {key!r} must have shape ({EXPECTED_DIM},), got {shape!r}")

    for key in CAMERA_KEYS:
        if key not in (info.get("features") or {}):
            report.error(f"Missing camera feature {key!r} in meta/info.json")
        elif not _is_video_feature(info, key):
            report.error(f"Camera feature {key!r} must be stored as dtype='video'")


def _validate_stats(stats: dict[str, Any], report: ValidationReport) -> None:
    if not stats:
        return
    for key in VECTOR_KEYS:
        if key not in stats:
            report.error(f"meta/stats.json is missing statistics for {key!r}")
            continue
        for stat_name in ("mean", "std", "min", "max"):
            value = np.asarray(stats[key].get(stat_name))
            if value.shape != (EXPECTED_DIM,):
                report.error(f"Stats for {key!r}.{stat_name} must have shape ({EXPECTED_DIM},), got {value.shape}")
            elif not np.isfinite(value).all():
                report.error(f"Stats for {key!r}.{stat_name} contain NaN or Inf")


def _validate_tasks(tasks: list[dict[str, Any]], report: ValidationReport) -> dict[int, str]:
    task_map: dict[int, str] = {}
    for row in tasks:
        task_index = row.get("task_index")
        task = row.get("task")
        if not isinstance(task_index, int):
            report.error(f"Task row has invalid task_index: {row!r}")
            continue
        if not isinstance(task, str) or not task:
            report.error(f"Task {task_index} has invalid task text: {task!r}")
            continue
        if task_index in task_map:
            report.error(f"Duplicate task_index in meta/tasks.jsonl: {task_index}")
            continue
        task_map[task_index] = task
    return task_map


def _episode_indices(episodes: list[dict[str, Any]], all_episodes: bool, max_episodes: int) -> list[int]:
    indices = [int(row["episode_index"]) for row in episodes if isinstance(row.get("episode_index"), int)]
    indices = sorted(indices)
    if all_episodes or max_episodes <= 0:
        return indices
    return indices[:max_episodes]


def _validate_episode_metadata(episodes: list[dict[str, Any]], task_map: dict[int, str], report: ValidationReport) -> None:
    seen = set()
    for row in episodes:
        episode_index = row.get("episode_index")
        length = row.get("length")
        tasks = row.get("tasks")
        if not isinstance(episode_index, int) or episode_index < 0:
            report.error(f"Episode row has invalid episode_index: {row!r}")
            continue
        if episode_index in seen:
            report.error(f"Duplicate episode_index in meta/episodes.jsonl: {episode_index}")
        seen.add(episode_index)
        if not isinstance(length, int) or length <= 0:
            report.error(f"Episode {episode_index} has invalid length: {length!r}")
        if not isinstance(tasks, list) or not tasks:
            report.error(f"Episode {episode_index} has no task metadata")
            continue
        missing_tasks = [task for task in tasks if task not in task_map.values()]
        if missing_tasks:
            report.error(f"Episode {episode_index} references unknown task text(s): {missing_tasks}")


def _read_episode_table(path: Path, report: ValidationReport) -> Any | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report.error("pyarrow is required to validate episode parquet files; install the dataset extra")
        return None

    try:
        return pq.read_table(path).to_pandas()
    except Exception as exc:
        report.error(f"Could not read parquet {_relative(path, report.root)}: {exc}")
        return None


def _validate_numeric_column(df: Any, key: str, episode_index: int, report: ValidationReport) -> None:
    values = np.asarray(df[key].tolist())
    if values.ndim != 2 or values.shape[1] != EXPECTED_DIM:
        report.error(f"Episode {episode_index} column {key!r} must be Nx{EXPECTED_DIM}, got {values.shape}")
        return
    if not np.isfinite(values).all():
        report.error(f"Episode {episode_index} column {key!r} contains NaN or Inf")
    if key == "action":
        grippers = values[:, [6, 13]]
        if ((grippers < 0.0) | (grippers > 1.0)).any():
            report.error(f"Episode {episode_index} action gripper values must stay in [0, 1]")
        joints = values[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
        if np.abs(joints).max(initial=0.0) > 2 * math.pi:
            report.warning(f"Episode {episode_index} action joint values exceed +/-2pi; verify units and calibration")
    if key == "observation.state":
        grippers = values[:, [6, 13]]
        if ((grippers < 0.0) | (grippers > 1.0)).any():
            report.error(f"Episode {episode_index} state gripper values must stay in [0, 1]")


def _validate_data_file(
    df: Any,
    episode_index: int,
    expected_length: int,
    task_map: dict[int, str],
    report: ValidationReport,
) -> None:
    for key in REQUIRED_DATA_KEYS:
        if key not in df.columns:
            report.error(f"Episode {episode_index} data is missing column {key!r}")
    if any(key not in df.columns for key in REQUIRED_DATA_KEYS):
        return

    if len(df) != expected_length:
        report.error(f"Episode {episode_index} length mismatch: metadata={expected_length}, parquet={len(df)}")
    if set(df["episode_index"].dropna().astype(int).unique()) != {episode_index}:
        report.error(f"Episode {episode_index} parquet contains another episode_index")
    if sorted(df["frame_index"].astype(int).tolist()) != list(range(len(df))):
        report.error(f"Episode {episode_index} frame_index must be contiguous from 0")
    timestamps = df["timestamp"].astype(float).to_numpy()
    if not np.isfinite(timestamps).all():
        report.error(f"Episode {episode_index} timestamp contains NaN or Inf")
    elif len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        report.error(f"Episode {episode_index} timestamp must be strictly increasing")

    invalid_tasks = sorted(set(df["task_index"].astype(int).tolist()) - set(task_map))
    if invalid_tasks:
        report.error(f"Episode {episode_index} parquet references unknown task_index values: {invalid_tasks}")

    for key in VECTOR_KEYS:
        _validate_numeric_column(df, key, episode_index, report)


def _validate_video_file(path: Path, expected_width: int | None, expected_height: int | None, report: ValidationReport) -> None:
    if not path.exists():
        report.error(f"Missing video file: {_relative(path, report.root)}")
        return
    if path.stat().st_size == 0:
        report.error(f"Video file is empty: {_relative(path, report.root)}")
        return

    try:
        import av
    except ImportError:
        report.warning("PyAV is not installed; video readability and size checks were skipped")
        return

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if expected_width is not None and stream.width != expected_width:
                report.error(f"{_relative(path, report.root)} width mismatch: expected {expected_width}, got {stream.width}")
            if expected_height is not None and stream.height != expected_height:
                report.error(
                    f"{_relative(path, report.root)} height mismatch: expected {expected_height}, got {stream.height}"
                )
            if stream.base_rate is not None and float(stream.base_rate) <= 0:
                report.error(f"{_relative(path, report.root)} has invalid FPS metadata")
    except Exception as exc:
        report.error(f"Could not read video {_relative(path, report.root)}: {exc}")


def validate_dataset(
    root: str | Path,
    *,
    all_episodes: bool = False,
    max_episodes: int = 5,
    check_videos: bool = True,
) -> ValidationReport:
    root = Path(root)
    report = ValidationReport(root=root)

    info = _load_json(root / "meta" / "info.json", report)
    stats = _load_json(root / "meta" / "stats.json", report)
    tasks = _load_jsonl(root / "meta" / "tasks.jsonl", report)
    episodes = _load_jsonl(root / "meta" / "episodes.jsonl", report)

    _validate_info(info, report)
    _validate_stats(stats, report)
    task_map = _validate_tasks(tasks, report)
    _validate_episode_metadata(episodes, task_map, report)

    report.total_episodes = len(episodes)
    report.total_frames = sum(row.get("length", 0) for row in episodes if isinstance(row.get("length"), int))

    expected_camera_size = {}
    for key in CAMERA_KEYS:
        shape = _feature_shape(info, key)
        expected_camera_size[key] = (shape[1], shape[0]) if shape and len(shape) >= 2 else (None, None)

    episode_by_index = {row["episode_index"]: row for row in episodes if isinstance(row.get("episode_index"), int)}
    selected = _episode_indices(episodes, all_episodes, max_episodes)
    report.sampled_episodes = len(selected)

    for episode_index in selected:
        row = episode_by_index[episode_index]
        data_file = _resolve_data_file(root, episode_index)
        if not data_file.exists():
            report.error(f"Missing data file: {_relative(data_file, root)}")
        else:
            df = _read_episode_table(data_file, report)
            if df is not None:
                _validate_data_file(df, episode_index, int(row["length"]), task_map, report)
                report.sampled_frames += len(df)

        if check_videos:
            for camera_key in CAMERA_KEYS:
                width, height = expected_camera_size[camera_key]
                _validate_video_file(_resolve_video_file(root, camera_key, episode_index), width, height, report)

    return report


def print_report(report: ValidationReport) -> None:
    status = "OK" if report.ok else "FAILED"
    print(f"X-trainer LeRobot v2.1 dataset validation: {status}")
    print(f"root: {report.root}")
    print(
        f"episodes: {report.total_episodes}, frames: {report.total_frames}, "
        f"sampled_episodes: {report.sampled_episodes}, sampled_frames: {report.sampled_frames}"
    )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an X-trainer LeRobot v2.1 dataset without modifying it.")
    parser.add_argument("--root", required=True, help="Dataset root containing meta/, data/, and videos/.")
    parser.add_argument("--all-episodes", action="store_true", help="Scan every episode instead of a small sample.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5,
        help="Maximum number of episodes to sample when --all-episodes is not set. Use 0 for all.",
    )
    parser.add_argument("--skip-videos", action="store_true", help="Skip video readability and shape checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_dataset(
        args.root,
        all_episodes=args.all_episodes,
        max_episodes=args.max_episodes,
        check_videos=not args.skip_videos,
    )
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
