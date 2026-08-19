"""Tests for the read-only LeRobot Dataset v2.1 metadata adapter."""

import hashlib

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pyarrow", reason="pyarrow is required (install lerobot[dataset])")
pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

from lerobot.datasets.v21 import LeRobotDatasetMetadataV21


def test_loads_xtrainer_metadata_without_changing_source(v21_xtrainer_dataset):
    root, before_hashes = v21_xtrainer_dataset

    metadata = LeRobotDatasetMetadataV21("test/xtrainer", root=root)

    assert metadata.fps == 10
    assert metadata.total_episodes == 2
    assert metadata.total_frames == 5
    assert metadata.camera_keys == [
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert metadata.features["action"]["shape"] == (14,)
    assert metadata.stats["action"]["mean"].shape == (14,)
    assert metadata.tasks == {0: "pick left", 1: "pick right"}
    assert str(metadata.get_data_file_path(1)) == "data/chunk-000/episode_000001.parquet"
    assert str(metadata.get_video_file_path(1, "observation.images.right_wrist")) == (
        "videos/chunk-000/observation.images.right_wrist/episode_000001.mp4"
    )

    after_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after_hashes == before_hashes


def test_rejects_non_v21_dataset(v21_xtrainer_dataset):
    root, _ = v21_xtrainer_dataset
    info_path = root / "meta" / "info.json"
    info_path.write_text(info_path.read_text(encoding="utf-8").replace("v2.1", "v3.0"), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected LeRobot Dataset v2.1"):
        LeRobotDatasetMetadataV21("test/xtrainer", root=root)
