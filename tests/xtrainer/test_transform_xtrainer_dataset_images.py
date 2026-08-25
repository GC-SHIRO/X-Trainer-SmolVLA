from __future__ import annotations

import numpy as np
import pytest

from tools.transform_xtrainer_dataset_images import CAMERA_KEYS, transform_frame


@pytest.mark.parametrize(
    ("camera_key", "expected"),
    [
        (CAMERA_KEYS[0], np.array([[1, 2, 3], [4, 5, 6]])),
        (CAMERA_KEYS[1], np.array([[3, 2, 1], [6, 5, 4]])),
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
