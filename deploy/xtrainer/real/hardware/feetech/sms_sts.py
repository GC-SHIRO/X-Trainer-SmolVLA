"""Small SMS/STS (Feetech protocol 0) transport used by X-trainer grippers.

The implementation intentionally covers only the operations used by the
X-trainer end effector: ping, two-byte reads, and position writes. It follows
the packet/register layout used by Dobot's X-trainer reference client, while
avoiding a model-number allowlist: the factory grippers identify themselves as
model 10760, not LeRobot's ``sts3215`` model 777.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol


_HEADER = b"\xff\xff"
_PING = 1
_READ = 2
_WRITE = 3
_MODEL_NUMBER_ADDRESS = 3
_ACCELERATION_ADDRESS = 41
_PRESENT_POSITION_ADDRESS = 56


class SerialLike(Protocol):
    is_open: bool

    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


class SmsStsProtocolError(RuntimeError):
    """Raised when an SMS/STS gripper returns an invalid status packet."""


class SmsStsGripperBus:
    """Protocol-0 single-servo transport with delayed ``pyserial`` import."""

    def __init__(
        self,
        *,
        port: str,
        motor_id: int,
        baudrate: int = 1_000_000,
        timeout_s: float = 0.1,
        serial_factory: Callable[..., SerialLike] | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= motor_id < 0xFE:
            raise ValueError("motor_id must be in [0, 253]")
        if baudrate <= 0 or timeout_s <= 0:
            raise ValueError("baudrate and timeout_s must be positive")
        self.port = port
        self.motor_id = motor_id
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self._serial_factory = serial_factory
        self._monotonic = monotonic_fn
        self._serial: SerialLike | None = None
        self.model_number: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and bool(getattr(self._serial, "is_open", True))

    def connect(self) -> int:
        if self.is_connected:
            assert self.model_number is not None
            return self.model_number

        serial_factory = self._serial_factory or self._default_serial_factory
        serial_port = serial_factory(port=self.port, baudrate=self.baudrate, timeout=self.timeout_s)
        self._serial = serial_port
        try:
            self._request(_PING)
            model_bytes = self.read_bytes(_MODEL_NUMBER_ADDRESS, 2)
            self.model_number = model_bytes[0] | (model_bytes[1] << 8)
            return self.model_number
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        serial_port, self._serial = self._serial, None
        self.model_number = None
        if serial_port is not None:
            serial_port.close()

    def read_position(self) -> int:
        data = self.read_bytes(_PRESENT_POSITION_ADDRESS, 2)
        raw = data[0] | (data[1] << 8)
        # SMS/STS encodes a negative position with bit 15 set. A physical
        # X-trainer gripper must report a non-negative absolute position.
        if raw & 0x8000:
            return -(raw & 0x7FFF)
        return raw

    def write_position(self, position: int, *, speed: int = 4096, acceleration: int = 0) -> None:
        if not 0 <= position <= 0xFFFF:
            raise ValueError("position must fit in an unsigned 16-bit value")
        if not 0 <= speed <= 0xFFFF:
            raise ValueError("speed must fit in an unsigned 16-bit value")
        if not 0 <= acceleration <= 0xFF:
            raise ValueError("acceleration must fit in an unsigned byte")
        payload = bytes(
            [
                _ACCELERATION_ADDRESS,
                acceleration,
                position & 0xFF,
                position >> 8,
                0,
                0,
                speed & 0xFF,
                speed >> 8,
            ]
        )
        self._request(_WRITE, payload)

    def read_bytes(self, address: int, length: int) -> bytes:
        if not 0 <= address <= 0xFF or not 1 <= length <= 0xFF:
            raise ValueError("invalid SMS/STS read range")
        return self._request(_READ, bytes([address, length]), response_length=length)

    def _request(self, instruction: int, parameters: bytes = b"", *, response_length: int = 0) -> bytes:
        serial_port = self._require_serial()
        length = len(parameters) + 2
        packet_without_checksum = bytes([self.motor_id, length, instruction]) + parameters
        packet = _HEADER + packet_without_checksum + bytes([_checksum(packet_without_checksum)])
        written = serial_port.write(packet)
        if written != len(packet):
            raise SmsStsProtocolError(f"incomplete write to {self.port}: {written}/{len(packet)} bytes")
        return self._read_status(response_length)

    def _read_status(self, response_length: int) -> bytes:
        packet = self._read_exact(6 + response_length)
        if packet[:2] != _HEADER:
            raise SmsStsProtocolError(f"invalid SMS/STS header from {self.port}: {packet!r}")
        motor_id, length, error = packet[2], packet[3], packet[4]
        if motor_id != self.motor_id:
            raise SmsStsProtocolError(f"unexpected SMS/STS motor id {motor_id}; expected {self.motor_id}")
        if length != response_length + 2 or len(packet) != length + 4:
            raise SmsStsProtocolError("unexpected SMS/STS status packet length")
        if packet[-1] != _checksum(packet[2:-1]):
            raise SmsStsProtocolError("SMS/STS status packet checksum mismatch")
        if error:
            raise SmsStsProtocolError(f"SMS/STS motor {self.motor_id} returned error {error}")
        return packet[5:-1]

    def _read_exact(self, size: int) -> bytes:
        serial_port = self._require_serial()
        deadline = self._monotonic() + self.timeout_s
        chunks = bytearray()
        while len(chunks) < size and self._monotonic() < deadline:
            received = serial_port.read(size - len(chunks))
            if received:
                chunks.extend(received)
        if len(chunks) != size:
            raise SmsStsProtocolError(
                f"timed out reading SMS/STS response from {self.port}: got {len(chunks)}/{size} bytes"
            )
        return bytes(chunks)

    def _require_serial(self) -> SerialLike:
        if not self.is_connected:
            raise ConnectionError("SMS/STS gripper is not connected")
        assert self._serial is not None
        return self._serial

    @staticmethod
    def _default_serial_factory(**kwargs: Any) -> SerialLike:
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depends on hardware host setup
            raise ImportError("pyserial is required for X-trainer Feetech grippers") from exc
        return serial.Serial(**kwargs)


def _checksum(payload: bytes) -> int:
    return (~sum(payload)) & 0xFF
