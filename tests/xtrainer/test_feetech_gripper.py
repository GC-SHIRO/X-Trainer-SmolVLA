import pytest

from deploy.xtrainer.real.hardware.feetech import XTrainerFeetechGripper, XTrainerFeetechGripperConfig
from deploy.xtrainer.real.hardware.feetech.sms_sts import SmsStsGripperBus, SmsStsProtocolError


def _status_packet(motor_id, data=b"", error=0):
    payload = bytes([motor_id, len(data) + 2, error]) + data
    return b"\xff\xff" + payload + bytes([(~sum(payload)) & 0xFF])


class FakeSerial:
    def __init__(self, responses):
        self.responses = list(responses)
        self.writes = []
        self.is_open = True
        self.closed = False

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def read(self, size=1):
        if not self.responses:
            return b""
        response = self.responses[0]
        result, remaining = response[:size], response[size:]
        if remaining:
            self.responses[0] = remaining
        else:
            self.responses.pop(0)
        return result

    def close(self):
        self.is_open = False
        self.closed = True


def test_sms_sts_bus_accepts_xtrainer_model_10760_and_uses_reference_registers():
    motor_id = 21
    fake_serial = FakeSerial(
        [
            _status_packet(motor_id),
            _status_packet(motor_id, bytes([0x08, 0x2A])),  # 10760, little endian
            _status_packet(motor_id, bytes([0x00, 0x08])),  # position 2048
            _status_packet(motor_id),
        ]
    )
    bus = SmsStsGripperBus(
        port="/dev/ttyUSB1",
        motor_id=motor_id,
        serial_factory=lambda **_kwargs: fake_serial,
    )

    assert bus.connect() == 10760
    assert bus.read_position() == 2048
    bus.write_position(2550)

    assert fake_serial.writes == [
        bytes([0xFF, 0xFF, 21, 2, 1, 0xE7]),
        bytes([0xFF, 0xFF, 21, 4, 2, 3, 2, 0xDF]),
        bytes([0xFF, 0xFF, 21, 4, 2, 56, 2, 0xAA]),
        bytes([0xFF, 0xFF, 21, 10, 3, 41, 0, 0xF6, 0x09, 0, 0, 0, 0x10, 0xA5]),
    ]


def test_xtrainer_gripper_maps_reference_servo_range_without_connect_motion():
    class FakeBus:
        is_connected = True

        def __init__(self, **_kwargs):
            self.positions = [2048, 3052]
            self.commands = []
            self.closed = False

        def connect(self):
            return 10760

        def read_position(self):
            return self.positions.pop(0)

        def write_position(self, value):
            self.commands.append(value)

        def close(self):
            self.closed = True

    gripper = XTrainerFeetechGripper(
        XTrainerFeetechGripperConfig(port="/dev/ttyUSB1", motor_id=21), bus_factory=FakeBus
    )

    gripper.connect()
    assert gripper.model_number == 10760
    assert gripper.read() == pytest.approx(0.0)
    assert gripper.read() == pytest.approx(1.0)
    gripper.write(0.5)

    assert gripper._bus.commands == [2550]


def test_connect_failure_closes_serial_port():
    motor_id = 21
    fake_serial = FakeSerial([_status_packet(motor_id, error=1)])
    bus = SmsStsGripperBus(
        port="/dev/ttyUSB1",
        motor_id=motor_id,
        serial_factory=lambda **_kwargs: fake_serial,
    )

    with pytest.raises(SmsStsProtocolError, match="returned error 1"):
        bus.connect()

    assert fake_serial.closed
    assert not bus.is_connected


def test_gripper_rejects_non_finite_commands():
    class FakeBus:
        is_connected = True

        def __init__(self, **_kwargs):
            pass

        def connect(self):
            return 10760

        def close(self):
            pass

    gripper = XTrainerFeetechGripper(
        XTrainerFeetechGripperConfig(port="/dev/ttyUSB1", motor_id=21), bus_factory=FakeBus
    )
    gripper.connect()

    with pytest.raises(ValueError, match="finite"):
        gripper.write(float("nan"))
