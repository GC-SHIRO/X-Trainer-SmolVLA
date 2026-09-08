import numpy as np
import pytest

from scripts.xtrainer.check_real_hardware import _move_gripper, _validate_args, parse_args


def test_hardware_check_requires_explicit_motion_confirmation():
    with pytest.raises(RuntimeError, match="pass --execute"):
        _validate_args(parse_args([]))

    _validate_args(parse_args(["--execute"]))


def test_hardware_check_defaults_to_xtrainer_frequency():
    args = parse_args([])

    assert args.control_hz == pytest.approx(30.0)
    assert np.isinf(args.max_gripper_delta)


def test_gripper_check_advances_in_bounded_steps():
    class FakeEnvironment:
        def __init__(self):
            self.commands = []

        def apply_action(self, action):
            applied = np.asarray(action, dtype=np.float64).copy()
            applied[6] = min(applied[6], self.commands[-1][6] + 0.2) if self.commands else min(applied[6], 0.2)
            self.commands.append(applied)
            return applied

    environment = FakeEnvironment()
    final = _move_gripper(environment, np.zeros(14), 6, 0.5, 0.0)

    assert final[6] == pytest.approx(0.5)
    assert [command[6] for command in environment.commands] == pytest.approx([0.2, 0.4, 0.5])
