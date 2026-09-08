import numpy as np
import pytest

from deploy.xtrainer.async_observation import observations_similar, validate_similarity_epsilon


def _payload(state=None, task="pick"):
    return {"state": np.zeros(14, dtype=np.float32) if state is None else state, "task": task}


def test_similarity_uses_arm_joints_but_protects_grippers_and_task():
    base = _payload()
    close = _payload(np.full(14, 0.001, dtype=np.float32))
    close["state"][[6, 13]] = 0
    assert observations_similar(close, base, epsilon=0.01)

    gripper_changed = _payload()
    gripper_changed["state"][6] = 0.1
    assert not observations_similar(gripper_changed, base, epsilon=0.01)
    assert not observations_similar(_payload(task="place"), base, epsilon=0.01)
    assert not observations_similar(base, base, epsilon=0)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_similarity_epsilon_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_similarity_epsilon(value)
