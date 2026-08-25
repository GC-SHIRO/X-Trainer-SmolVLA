from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.transform_xtrainer_dataset_images import CAMERA_KEYS, _video_paths, transform_frame


@pytest.mark.parametrize(
    ("camera_key", "expected"),
    [
        (CAMERA_KEYS[0], np.array([[1, 2, 3], [4, 5, 6]])),
        (CAMERA_KEYS[1], np.array([[1, 2, 3], [4, 5, 6]])),
        (CAMERA_KEYS[2], np.array([[6, 5, 4], [3, 2, 1]])),
    ],
)
def test_transform_frame_applies_the_camera_orientation_rule(camera_key, expected):
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)[..., None]

    transformed = transform_frame(camera_key, image)

    np.testing.assert_array_equal(transformed[..., 0], expected)
    assert transformed.flags.c_contiguous


def test_transform_frame_rejects_unknown_camera_key():
    with pytest.raises(KeyError, match="Unsupported"):
        transform_frame("observation.images.unknown", np.zeros((2, 2, 3), dtype=np.uint8))


def test_video_paths_aligns_cameras_by_chunk_and_episode(tmp_path):
    for camera_key in CAMERA_KEYS:
        camera_dir = tmp_path / "videos" / "chunk-000" / camera_key
        camera_dir.mkdir(parents=True)
        for episode_index in (0, 1):
            (camera_dir / f"episode_{episode_index:06d}.mp4").touch()

    paths = {camera_key: _video_paths(tmp_path, camera_key) for camera_key in CAMERA_KEYS}

    expected = {Path("chunk-000/episode_000000.mp4"), Path("chunk-000/episode_000001.mp4")}
    assert all(set(camera_paths) == expected for camera_paths in paths.values())
