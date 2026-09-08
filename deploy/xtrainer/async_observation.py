"""Small helpers shared by the X-Trainer latest-observation transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

JOINT_INDICES = np.array([*range(6), *range(7, 13)])
GRIPPER_INDICES = np.array([6, 13])


@dataclass(frozen=True)
class PendingObservation:
    observation_id: int
    observation_timestep: int
    must_go: bool
    payload: dict[str, Any]
    session_id: str


def validate_similarity_epsilon(value: Any) -> float | None:
    if value is None:
        return None
    epsilon = float(value)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("observation_similarity_epsilon must be finite and non-negative")
    return epsilon


def observations_similar(candidate: dict[str, Any], previous: dict[str, Any], epsilon: float | None) -> bool:
    """Compare X-Trainer state conservatively; images are intentionally not compared."""

    if epsilon is None or epsilon == 0:
        return False
    if candidate.get("task") != previous.get("task"):
        return False
    candidate_state = np.asarray(candidate.get("state"), dtype=np.float32)
    previous_state = np.asarray(previous.get("state"), dtype=np.float32)
    if candidate_state.shape != (14,) or previous_state.shape != (14,):
        return False
    if not np.array_equal(candidate_state[GRIPPER_INDICES], previous_state[GRIPPER_INDICES]):
        return False
    return bool(np.linalg.norm(candidate_state[JOINT_INDICES] - previous_state[JOINT_INDICES]) < epsilon)
