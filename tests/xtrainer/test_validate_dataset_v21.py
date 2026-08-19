import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "xtrainer" / "validate_dataset_v21.py"
spec = importlib.util.spec_from_file_location("validate_dataset_v21", SCRIPT_PATH)
validate_dataset_v21 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = validate_dataset_v21
spec.loader.exec_module(validate_dataset_v21)


CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_dataset(root: Path, *, include_videos: bool = True) -> Path:
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "action": {"dtype": "float32", "shape": [14], "names": None},
        **{
            key: {"dtype": "video", "shape": [8, 10, 3], "names": ["height", "width", "channels"]}
            for key in CAMERA_KEYS
        },
    }
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "fps": 20,
            "features": features,
            "total_episodes": 2,
            "total_frames": 6,
            "total_tasks": 1,
        },
    )
    vector_stats = {
        "mean": [0.0] * 14,
        "std": [1.0] * 14,
        "min": [-1.0] * 14,
        "max": [1.0] * 14,
        "count": [6],
    }
    _write_json(
        root / "meta" / "stats.json",
        {"observation.state": vector_stats, "action": vector_stats},
    )
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "pick up the object"}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick up the object"], "length": 3},
            {"episode_index": 1, "tasks": ["pick up the object"], "length": 3},
        ],
    )

    for episode_index in range(2):
        state = np.zeros((3, 14), dtype=np.float32)
        action = np.zeros((3, 14), dtype=np.float32)
        action[:, 6] = 0.5
        action[:, 13] = 0.5
        df = pd.DataFrame(
            {
                "observation.state": list(state),
                "action": list(action),
                "timestamp": [0.0, 0.05, 0.10],
                "frame_index": [0, 1, 2],
                "episode_index": [episode_index] * 3,
                "index": [episode_index * 3 + i for i in range(3)],
                "task_index": [0, 0, 0],
            }
        )
        data_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)

        if include_videos:
            for camera_key in CAMERA_KEYS:
                video_path = root / "videos" / "chunk-000" / camera_key / f"episode_{episode_index:06d}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"not-a-real-video-but-present")
    return root


def test_valid_fixture_passes_when_video_probe_is_skipped(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=True)

    report = validate_dataset_v21.validate_dataset(root, check_videos=False)

    assert report.ok
    assert report.total_episodes == 2
    assert report.total_frames == 6
    assert report.sampled_frames == 6


def test_missing_camera_fails(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=False)

    report = validate_dataset_v21.validate_dataset(root, check_videos=True)

    assert not report.ok
    assert any("Missing video file" in error for error in report.errors)


def test_wrong_action_dimension_fails(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=False)
    path = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(path)
    df["action"] = list(np.zeros((3, 13), dtype=np.float32))
    df.to_parquet(path, index=False)

    report = validate_dataset_v21.validate_dataset(root, check_videos=False)

    assert not report.ok
    assert any("column 'action' must be Nx14" in error for error in report.errors)


def test_invalid_task_index_fails(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=False)
    path = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(path)
    df["task_index"] = [99, 99, 99]
    df.to_parquet(path, index=False)

    report = validate_dataset_v21.validate_dataset(root, check_videos=False)

    assert not report.ok
    assert any("unknown task_index" in error for error in report.errors)


def test_nan_state_fails(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=False)
    path = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(path)
    state = np.zeros((3, 14), dtype=np.float32)
    state[1, 2] = np.nan
    df["observation.state"] = list(state)
    df.to_parquet(path, index=False)

    report = validate_dataset_v21.validate_dataset(root, check_videos=False)

    assert not report.ok
    assert any("observation.state" in error and "NaN or Inf" in error for error in report.errors)


def test_missing_video_fails(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=True)
    (root / "videos" / "chunk-000" / CAMERA_KEYS[0] / "episode_000000.mp4").unlink()

    report = validate_dataset_v21.validate_dataset(root, check_videos=True)

    assert not report.ok
    assert any("Missing video file" in error for error in report.errors)


def test_validation_does_not_modify_source_files(tmp_path):
    root = _make_dataset(tmp_path / "dataset", include_videos=False)
    tracked = [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.jsonl",
        root / "meta" / "episodes.jsonl",
        root / "data" / "chunk-000" / "episode_000000.parquet",
    ]
    before = {path: path.read_bytes() for path in tracked}

    validate_dataset_v21.validate_dataset(root, check_videos=False)

    after = {path: path.read_bytes() for path in tracked}
    assert after == before
