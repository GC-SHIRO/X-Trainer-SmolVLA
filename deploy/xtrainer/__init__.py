"""X-trainer real-robot deployment transport.

This package exposes only the LAN transport contract used by X-trainer
deployment. Hardware and policy model imports stay in subpackages so training,
serving, and transport tests do not need real-robot dependencies.
"""

from .msgpack_numpy import ProtocolError, dumps, loads
from .websocket_client_policy import XTrainerWebSocketPolicyClient
from .websocket_policy_server import XTrainerWebSocketPolicyServer

__all__ = [
    "ProtocolError",
    "XTrainerWebSocketPolicyClient",
    "XTrainerWebSocketPolicyServer",
    "dumps",
    "loads",
]
