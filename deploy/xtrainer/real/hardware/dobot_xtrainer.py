"""Dobot TCP helpers for X-trainer dual-arm deployment."""

from __future__ import annotations

import math
import re
import socket
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


DEFAULT_DASHBOARD_PORT = 29999
DEFAULT_MOTION_PORT = 30003
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_SERVO_J_TIME_S = 0.03
DEFAULT_SERVO_GAIN = 500
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class DobotProtocolError(RuntimeError):
    """Raised when the Dobot controller returns an error or malformed data."""


class SocketLike(Protocol):
    def settimeout(self, timeout: float) -> None: ...

    def connect(self, address: tuple[str, int]) -> None: ...

    def sendall(self, data: bytes) -> None: ...

    def recv(self, bufsize: int) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DobotResponse:
    error_id: int | None
    payload: str
    command: str | None
    raw: str


@dataclass(frozen=True)
class DobotArmConfig:
    ip: str
    dashboard_port: int = DEFAULT_DASHBOARD_PORT
    motion_port: int = DEFAULT_MOTION_PORT
    timeout_s: float = DEFAULT_TIMEOUT_S
    joint_unit: str = "rad"


def parse_dobot_response(raw: str) -> DobotResponse:
    """Parse a Dobot response such as ``0,{...},Command();``.

    Some Dobot commands only return a plain acknowledgement string. When an
    error id is present, it must be zero for the command to be considered
    successful.
    """

    text = raw.strip().strip("\x00")
    command = None
    if "," in text:
        prefix, rest = text.split(",", 1)
        try:
            error_id = int(prefix)
        except ValueError:
            error_id = None
            payload = text
        else:
            payload = rest
            if "," in rest:
                payload, command = rest.rsplit(",", 1)
                command = command.strip() or None
            return DobotResponse(error_id=error_id, payload=payload.strip(), command=command, raw=text)
    return DobotResponse(error_id=None, payload=text, command=command, raw=text)


def _extract_joint_values(response: DobotResponse) -> np.ndarray:
    values = [float(match.group(0)) for match in _FLOAT_RE.finditer(response.payload)]
    if len(values) < 6:
        raise DobotProtocolError(f"expected 6 joint values in Dobot response, got {response.raw!r}")
    return np.asarray(values[:6], dtype=np.float64)


class DobotTcpClient:
    """Small synchronous TCP client for one Dobot endpoint."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        socket_factory=socket.socket,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.socket_factory = socket_factory
        self.sock: SocketLike | None = None

    @property
    def is_connected(self) -> bool:
        return self.sock is not None

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = self.socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout_s)
            sock.connect((self.host, self.port))
        except BaseException:
            sock.close()
            raise
        self.sock = sock

    def close(self) -> None:
        sock, self.sock = self.sock, None
        if sock is not None:
            sock.close()

    def request(self, command: str, *, check_ack: bool = True) -> DobotResponse:
        if self.sock is None:
            raise ConnectionError("Dobot TCP client is not connected")
        wire_command = command if command.endswith("\n") else f"{command}\n"
        self.sock.sendall(wire_command.encode("ascii"))
        raw = self.sock.recv(4096).decode("ascii", errors="replace")
        response = parse_dobot_response(raw)
        if check_ack and response.error_id not in (None, 0):
            raise DobotProtocolError(f"Dobot command failed: {response.raw}")
        return response


@dataclass
class XTrainerDobotArm:
    """Dobot arm wrapper with separate Dashboard and motion TCP sockets."""

    config: DobotArmConfig
    socket_factory: object = socket.socket
    dashboard: DobotTcpClient = field(init=False)
    motion: DobotTcpClient = field(init=False)

    def __post_init__(self) -> None:
        self.dashboard = DobotTcpClient(
            self.config.ip,
            self.config.dashboard_port,
            timeout_s=self.config.timeout_s,
            socket_factory=self.socket_factory,
        )
        self.motion = DobotTcpClient(
            self.config.ip,
            self.config.motion_port,
            timeout_s=self.config.timeout_s,
            socket_factory=self.socket_factory,
        )

    def connect(self) -> None:
        opened: list[DobotTcpClient] = []
        try:
            self.dashboard.connect()
            opened.append(self.dashboard)
            self.motion.connect()
            opened.append(self.motion)
        except BaseException:
            for client in reversed(opened):
                client.close()
            raise

    def close(self) -> None:
        self.motion.close()
        self.dashboard.close()

    def enable(self) -> None:
        self.dashboard.request("EnableRobot()")

    def disable(self) -> None:
        self.dashboard.request("DisableRobot()")

    def clear_error(self) -> None:
        self.dashboard.request("ClearError()")

    def read_joints(self) -> np.ndarray:
        response = self.dashboard.request("GetAngle()")
        joints_deg = _extract_joint_values(response)
        if not np.all(np.isfinite(joints_deg)):
            raise DobotProtocolError(f"Dobot returned non-finite joint values: {response.raw}")
        return np.deg2rad(joints_deg) if self.config.joint_unit == "rad" else joints_deg

    def move_joints(self, joints: np.ndarray | list[float]) -> None:
        array = np.asarray(joints, dtype=np.float64)
        if array.shape != (6,):
            raise ValueError(f"Dobot joint command must have shape (6,), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("Dobot joint command must contain only finite values")
        values = np.rad2deg(array) if self.config.joint_unit == "rad" else array
        if np.any(np.abs(values) > 360.0 * 4):
            raise ValueError("Dobot joint command is outside a plausible joint range")
        self.motion.request(make_joint_command(values))

    @classmethod
    def from_parts(
        cls,
        *,
        ip: str,
        dashboard_port: int = DEFAULT_DASHBOARD_PORT,
        motion_port: int = DEFAULT_MOTION_PORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        joint_unit: str = "rad",
        socket_factory=socket.socket,
    ) -> "XTrainerDobotArm":
        if joint_unit not in ("rad", "deg"):
            raise ValueError(f"joint_unit must be 'rad' or 'deg', got {joint_unit!r}")
        config = DobotArmConfig(
            ip=ip,
            dashboard_port=dashboard_port,
            motion_port=motion_port,
            timeout_s=timeout_s,
            joint_unit=joint_unit,
        )
        return cls(config=config, socket_factory=socket_factory)


def split_xtrainer_arm_action(action: np.ndarray | list[float], side: str) -> np.ndarray:
    """Extract the 6-DOF Dobot command from a 14-D X-trainer action."""

    array = np.asarray(action, dtype=np.float64)
    if array.shape != (14,):
        raise ValueError(f"X-trainer action must have shape (14,), got {array.shape}")
    if side == "left":
        return array[:6]
    if side == "right":
        return array[7:13]
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def make_joint_command(joints_deg: np.ndarray | list[float]) -> str:
    """Build the reference X-trainer ``ServoJ`` command from joint degrees."""

    array = np.asarray(joints_deg, dtype=np.float64)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError("joints_deg must be a finite 6-vector")
    args = ",".join(f"{value:.6f}" for value in array)
    return f"ServoJ({args},{DEFAULT_SERVO_J_TIME_S:.6f},gain={DEFAULT_SERVO_GAIN})"


def degrees_close(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-6) -> bool:
    return bool(np.allclose(np.rad2deg(a), b, atol=atol))


def radians(values: list[float]) -> np.ndarray:
    return np.asarray([math.radians(value) for value in values], dtype=np.float64)
