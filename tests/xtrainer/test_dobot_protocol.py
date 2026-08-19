import numpy as np
import pytest

from deploy.xtrainer.real.hardware import DEFAULT_XTRAINER_CAMERA_CONFIGS
from deploy.xtrainer.real.hardware.dobot_xtrainer import (
    DobotProtocolError,
    XTrainerDobotArm,
    make_joint_command,
    parse_dobot_response,
    split_xtrainer_arm_action,
)


class FakeSocket:
    created = []
    fail_connect_after = None

    def __init__(self, *_args):
        self.sent = []
        self.closed = False
        self.timeout = None
        self.responses = []
        FakeSocket.created.append(self)

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, _address):
        if FakeSocket.fail_connect_after is not None and len(FakeSocket.created) > FakeSocket.fail_connect_after:
            raise OSError("mock connect failed")

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, _bufsize):
        if self.responses:
            return self.responses.pop(0)
        return b"0,{},Mock();"

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_socket():
    FakeSocket.created = []
    FakeSocket.fail_connect_after = None


def test_parse_dobot_response_with_ack_and_payload():
    response = parse_dobot_response("0,{1,2,3,4,5,6},GetAngle();")

    assert response.error_id == 0
    assert response.payload == "{1,2,3,4,5,6}"
    assert response.command == "GetAngle();"


def test_parse_dobot_nonzero_ack_raises_from_client():
    arm = XTrainerDobotArm.from_parts(ip="192.168.5.1", socket_factory=FakeSocket)
    arm.connect()
    FakeSocket.created[0].responses.append(b"-1,{},EnableRobot();")

    with pytest.raises(DobotProtocolError, match="Dobot command failed"):
        arm.enable()


def test_dobot_read_joints_parses_degrees_as_radians():
    arm = XTrainerDobotArm.from_parts(ip="192.168.5.1", socket_factory=FakeSocket)
    arm.connect()
    FakeSocket.created[0].responses.append(b"0,{0,90,180,-90,45,-45},GetAngle();")

    joints = arm.read_joints()

    np.testing.assert_allclose(joints, np.deg2rad([0, 90, 180, -90, 45, -45]))


def test_dobot_read_joints_malformed_response_does_not_return_zeros():
    arm = XTrainerDobotArm.from_parts(ip="192.168.5.1", socket_factory=FakeSocket)
    arm.connect()
    FakeSocket.created[0].responses.append(b"0,{bad},GetAngle();")

    with pytest.raises(DobotProtocolError, match="expected 6 joint values"):
        arm.read_joints()


def test_dobot_move_joints_sends_motion_command():
    arm = XTrainerDobotArm.from_parts(ip="192.168.5.1", socket_factory=FakeSocket)
    arm.connect()

    arm.move_joints(np.deg2rad([1, 2, 3, 4, 5, 6]))

    assert FakeSocket.created[1].sent[-1] == (
        b"ServoJ(1.000000,2.000000,3.000000,4.000000,5.000000,6.000000,0.030000,gain=500)\n"
    )


def test_dobot_partial_connection_failure_closes_opened_socket():
    FakeSocket.fail_connect_after = 1
    arm = XTrainerDobotArm.from_parts(ip="192.168.5.1", socket_factory=FakeSocket)

    with pytest.raises(OSError):
        arm.connect()

    assert FakeSocket.created[0].closed is True


def test_split_xtrainer_arm_action_uses_fixed_indices():
    action = np.arange(14, dtype=np.float32)

    np.testing.assert_array_equal(split_xtrainer_arm_action(action, "left"), np.arange(6))
    np.testing.assert_array_equal(split_xtrainer_arm_action(action, "right"), np.arange(7, 13))


def test_make_joint_command_rejects_bad_shape():
    with pytest.raises(ValueError):
        make_joint_command([1, 2, 3])


def test_make_joint_command_matches_reference_servoj_format():
    assert make_joint_command([1, 2, 3, 4, 5, 6]) == (
        "ServoJ(1.000000,2.000000,3.000000,4.000000,5.000000,6.000000,0.030000,gain=500)"
    )


def test_default_realsense_serials_map_to_observation_keys():
    assert DEFAULT_XTRAINER_CAMERA_CONFIGS["top"].serial == "409122273405"
    assert DEFAULT_XTRAINER_CAMERA_CONFIGS["top"].observation_key == "observation.images.top"
    assert DEFAULT_XTRAINER_CAMERA_CONFIGS["left_wrist"].serial == "412622272997"
    assert DEFAULT_XTRAINER_CAMERA_CONFIGS["left_wrist"].observation_key == "observation.images.left_wrist"
    assert DEFAULT_XTRAINER_CAMERA_CONFIGS["right_wrist"].serial == "412622271417"
    assert DEFAULT_XTRAINER_CAMERA_CONFIGS["right_wrist"].observation_key == "observation.images.right_wrist"
