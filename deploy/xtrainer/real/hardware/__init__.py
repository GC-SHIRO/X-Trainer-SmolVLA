"""Hardware drivers used by X-trainer real-robot deployment.

Imports in this package are safe on machines without Dobot, RealSense, or
serial hardware dependencies installed. Device-specific SDKs are imported only
when a concrete driver connects to hardware.
"""

from .dobot_xtrainer import DobotProtocolError, XTrainerDobotArm, parse_dobot_response
from .realsense_camera import (
    DEFAULT_XTRAINER_CAMERA_CONFIGS,
    XTrainerRealSenseCamera,
    XTrainerRealSenseCameraConfig,
)

__all__ = [
    "DEFAULT_XTRAINER_CAMERA_CONFIGS",
    "DobotProtocolError",
    "XTrainerDobotArm",
    "XTrainerRealSenseCamera",
    "XTrainerRealSenseCameraConfig",
    "parse_dobot_response",
]
