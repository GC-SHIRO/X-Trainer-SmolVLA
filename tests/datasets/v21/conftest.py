"""Fixtures for read-only LeRobot Dataset v2.1 adapter tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def v21_xtrainer_dataset(tmp_path):
    """Create a two-episode, three-camera v2.1 X-trainer fixture."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "xtrainer_v21"
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "action": {"dtype": "float32", "shape": [14], "names": None},
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.images.top": {"dtype": "video", "shape": [3, 224, 224], "names": None},
        "observation.images.left_wrist": {"dtype": "video", "shape": [3, 224, 224], "names": None},
        "observation.images.right_wrist": {"dtype": "video", "shape": [3, 224, 224], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    stats = {
        "observation.state": {"mean": [0.0] * 14, "std": [1.0] * 14},
        "action": {"mean": [0.0] * 14, "std": [1.0] * 14},
    }
    (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        '\n'.join((json.dumps({"task_index": 0, "task": "pick left"}), json.dumps({"task_index": 1, "task": "pick right"})))
        + '\n',
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        '\n'.join((json.dumps({"episode_index": 0, "length": 3, "tasks": ["pick left"]}), json.dumps({"episode_index": 1, "length": 2, "tasks": ["pick right"]})))
        + '\n',
        encoding="utf-8",
    )

    cameras = ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist")
    for episode_index, length in ((0, 3), (1, 2)):
        base = episode_index * 100
        states = (np.arange(length * 14, dtype=np.float32).reshape(length, 14) + base).tolist()
        actions = (np.arange(length * 14, dtype=np.float32).reshape(length, 14) + base + 0.5).tolist()
        table = pa.table(
            {
                "observation.state": states,
                "action": actions,
                "timestamp": [frame / 10 for frame in range(length)],
                "task_index": [episode_index] * length,
                "episode_index": [episode_index] * length,
                "frame_index": list(range(length)),
                "index": list(range(base, base + length)),
            }
        )
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        for camera in cameras:
            video_path = root / "videos" / "chunk-000" / camera / f"episode_{episode_index:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fixture video; decoder is mocked")

    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    return root, hashes
