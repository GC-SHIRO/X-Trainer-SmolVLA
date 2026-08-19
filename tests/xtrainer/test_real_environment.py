import numpy as np
import pytest

from deploy.xtrainer.real import XTrainerRealEnvironment as PublicXTrainerRealEnvironment
from deploy.xtrainer.real.environment import (
    IMAGE_KEYS,
    STATE_KEY,
    TASK_KEY,
    XTrainerRealEnvironment,
    XTrainerSafetyConfig,
)
from deploy.xtrainer.real.hardware.realsense_camera import XTrainerRealSenseCameraConfig


class MockArm:
    def __init__(self, joints):
        self.joints = np.asarray(joints, dtype=np.float32)
        self.commands = []
        self.connected = False
        self.closed = False
        self.enabled = False
        self.disabled = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.disabled = True
        self.enabled = False

    def read_joints(self):
        return self.joints.copy()

    def move_joints(self, joints):
        self.commands.append(np.asarray(joints, dtype=np.float32))
        self.joints = self.commands[-1]


class MockGripper:
    def __init__(self, value):
        self.value = float(value)
        self.commands = []
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def read(self):
        return self.value

    def write(self, value):
        self.commands.append(float(value))
        self.value = float(value)


class MockCamera:
    def __init__(self, name, key):
        self.config = XTrainerRealSenseCameraConfig(name=name, serial=f"{name}-serial", observation_key=key)
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def read_rgb(self):
        return np.zeros((4, 5, 3), dtype=np.uint8)


class BadShapeCamera(MockCamera):
    def read_rgb(self):
        return np.zeros((4, 5), dtype=np.uint8)


def make_env(**kwargs):
    cameras = {
        "top": MockCamera("top", "observation.images.top"),
        "left_wrist": MockCamera("left_wrist", "observation.images.left_wrist"),
        "right_wrist": MockCamera("right_wrist", "observation.images.right_wrist"),
    }
    sleep_fn = kwargs.pop("sleep_fn", lambda _seconds: None)
    return XTrainerRealEnvironment(
        left_arm=MockArm(np.arange(6, dtype=np.float32)),
        right_arm=MockArm(np.arange(7, 13, dtype=np.float32)),
        left_gripper=MockGripper(0.25),
        right_gripper=MockGripper(0.75),
        cameras=cameras,
        task="pick",
        sleep_fn=sleep_fn,
        **kwargs,
    )


def test_observation_fields_dimensions_and_camera_mapping():
    env = make_env()

    obs = env.reset()

    assert obs[STATE_KEY].shape == (14,)
    assert obs[STATE_KEY].dtype == np.float32
    assert obs[TASK_KEY] == "pick"
    for key in IMAGE_KEYS:
        assert obs[key].shape == (4, 5, 3)
        assert obs[key].dtype == np.uint8


def test_observation_rejects_missing_or_bad_camera_mapping():
    env = make_env()
    env.cameras.pop("right_wrist")

    with pytest.raises(KeyError):
        env.reset()

    env = make_env()
    env.cameras["top"] = BadShapeCamera("top", "observation.images.top")
    with pytest.raises(ValueError, match="HxWx3"):
        env.reset()


def test_package_exports_public_environment_interface():
    assert PublicXTrainerRealEnvironment is XTrainerRealEnvironment
    for method_name in ("reset", "get_observation", "apply_action", "close"):
        assert callable(getattr(PublicXTrainerRealEnvironment, method_name))


def test_apply_action_splits_left_right_arms_and_grippers_by_fixed_indices():
    env = make_env(
        safety=XTrainerSafetyConfig(
            max_joint_delta_rad=100.0,
            max_gripper_delta=1.0,
            gripper_update_threshold=0.0,
        )
    )
    env.reset()
    action = np.arange(14, dtype=np.float32) / 10.0
    action[13] = 0.8

    applied = env.apply_action(action)

    np.testing.assert_allclose(env.left_arm.commands[-1], action[:6])
    np.testing.assert_allclose(env.right_arm.commands[-1], action[7:13])
    assert env.left_gripper.commands[-1] == pytest.approx(float(action[6]))
    assert env.right_gripper.commands[-1] == pytest.approx(float(action[13]))
    np.testing.assert_allclose(applied, action)


def test_apply_action_rejects_bad_shape_and_non_finite():
    env = make_env()
    env.reset()

    with pytest.raises(ValueError, match="shape"):
        env.apply_action(np.zeros(13))
    bad = np.zeros(14)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        env.apply_action(bad)


def test_apply_action_enforces_final_safety_limits():
    env = make_env(
        safety=XTrainerSafetyConfig(
            max_joint_delta_rad=0.1,
            max_gripper_delta=0.05,
            joint_position_limit_rad=(-100.0, 100.0),
        )
    )
    env.reset()
    action = np.full(14, 100.0, dtype=np.float32)

    applied = env.apply_action(action)

    current = np.concatenate([np.arange(6), [0.25], np.arange(7, 13), [0.75]]).astype(np.float32)
    np.testing.assert_allclose(applied[np.r_[0:6, 7:13]], current[np.r_[0:6, 7:13]] + 0.1)
    assert applied[6] == pytest.approx(0.30)
    assert applied[13] == pytest.approx(0.80)


def test_gripper_update_threshold_skips_small_changes():
    env = make_env(safety=XTrainerSafetyConfig(max_joint_delta_rad=1.0, gripper_update_threshold=0.1))
    env.reset()
    action = env._last_state.copy()
    action[6] += 0.05
    action[13] -= 0.05

    env.apply_action(action)

    assert env.left_gripper.commands == []
    assert env.right_gripper.commands == []


def test_close_is_idempotent_and_cleans_all_resources():
    env = make_env()
    env.reset()
    env.enable_arms()

    env.close()
    env.close()

    assert env.left_arm.closed
    assert env.right_arm.closed
    assert env.left_arm.disabled
    assert env.right_arm.disabled
    assert env.left_gripper.closed
    assert env.right_gripper.closed
    assert all(camera.closed for camera in env.cameras.values())


def test_reset_after_close_reconnects_environment():
    env = make_env()
    env.reset()
    env.close()

    env.reset()

    assert env._closed is False
    assert env.left_arm.connected
    assert env.right_arm.connected


def test_partial_connection_failure_closes_opened_resources():
    env = make_env()

    def fail_connect():
        raise RuntimeError("camera failed")

    env.cameras["left_wrist"].connect = fail_connect

    with pytest.raises(RuntimeError, match="camera failed"):
        env.reset()

    assert env.left_arm.closed
    assert env.right_arm.closed
    assert env.left_gripper.closed
    assert env.right_gripper.closed
    assert env.cameras["top"].closed


def test_partial_arm_enable_failure_disables_already_enabled_arm():
    env = make_env()
    env.reset()

    def fail_enable():
        raise RuntimeError("right arm enable failed")

    env.right_arm.enable = fail_enable

    with pytest.raises(RuntimeError, match="right arm enable failed"):
        env.enable_arms()

    assert env.left_arm.disabled
    assert env._enabled_arms == []


def test_close_failure_does_not_skip_other_resource_cleanup():
    env = make_env()
    env.reset()
    env.enable_arms()

    def fail_close():
        raise RuntimeError("camera close failed")

    env.cameras["top"].close = fail_close

    env.close()

    assert env.left_arm.disabled
    assert env.right_arm.disabled
    assert env.left_arm.closed
    assert env.right_arm.closed
    assert env.left_gripper.closed
    assert env.right_gripper.closed


def test_smooth_reset_steps_with_control_period():
    sleeps = []
    env = make_env(sleep_fn=sleeps.append, safety=XTrainerSafetyConfig(ramp_step_rad=0.5, ramp_max_steps=20))
    env.reset()
    target = np.zeros(14, dtype=np.float32)
    target[0] = 2.0

    env.smooth_reset(target)

    assert len(env.left_arm.commands) > 1
    assert all(seconds == pytest.approx(1 / 20) for seconds in sleeps)


def test_apply_action_can_defer_pacing_to_external_control_loop():
    sleeps = []
    env = make_env(sleep_fn=sleeps.append)
    env.reset()

    env.apply_action(env._last_state, pace=False)

    assert sleeps == []
