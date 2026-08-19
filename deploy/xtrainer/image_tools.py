"""Image helpers shared by X-trainer real deployment code."""

from __future__ import annotations

from typing import Any

import numpy as np


def ensure_rgb_uint8(image: Any, *, name: str = "image") -> np.ndarray:
    """Return an HxWx3 uint8 RGB image or raise a clear validation error."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def validate_camera_observations(
    observations: dict[str, Any],
    required_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Validate and normalize a camera observation dict keyed by dataset fields."""

    missing = [key for key in required_keys if key not in observations]
    if missing:
        raise KeyError(f"missing camera observations: {missing}")
    return {key: ensure_rgb_uint8(observations[key], name=key) for key in required_keys}
