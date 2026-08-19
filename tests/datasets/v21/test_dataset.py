"""Tests for episode-local v2.1 data access and temporal padding."""

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pyarrow", reason="pyarrow is required (install lerobot[dataset])")
pytest.importorskip("av", reason="av is required (install lerobot[dataset])")
torch = pytest.importorskip("torch", reason="torch is required (install lerobot[dataset])")

from lerobot.datasets.v21 import LeRobotDatasetV21


def test_action_chunk_padding_task_and_video_timestamp_lookup(v21_xtrainer_dataset, monkeypatch):
    root, _ = v21_xtrainer_dataset
    decoded = []

    def fake_decode(path, timestamps, tolerance_s, backend, return_uint8):
        decoded.append((path, timestamps, tolerance_s, backend, return_uint8))
        return torch.full((len(timestamps), 3, 2, 2), len(decoded), dtype=torch.uint8)

    monkeypatch.setattr("lerobot.datasets.v21.dataset.decode_video_frames", fake_decode)
    dataset = LeRobotDatasetV21(
        "test/xtrainer",
        root=root,
        delta_timestamps={
            "action": [0.0, 0.1, 0.2],
            "observation.images.top": [0.0],
            "observation.images.left_wrist": [0.0],
            "observation.images.right_wrist": [0.0],
        },
    )

    last_item_of_first_episode = dataset[2]
    assert last_item_of_first_episode["action"].shape == (3, 14)
    assert last_item_of_first_episode["action_is_pad"].tolist() == [False, True, True]
    np.testing.assert_allclose(
        last_item_of_first_episode["action"][1:].numpy(),
        np.broadcast_to(last_item_of_first_episode["action"][0].numpy(), (2, 14)),
    )
    assert last_item_of_first_episode["task"] == "pick left"
    assert len(decoded) == 3
    assert all(call[1] == [0.2] for call in decoded)
    assert all("episode_000000.mp4" in str(call[0]) for call in decoded)

    first_item_of_second_episode = dataset[3]
    assert first_item_of_second_episode["task"] == "pick right"
    assert first_item_of_second_episode["observation.state"].shape == (14,)
    assert first_item_of_second_episode["observation.state"][0].item() == 100.0
    assert all(torch.isfinite(value).all() for value in first_item_of_second_episode.values() if isinstance(value, torch.Tensor))
