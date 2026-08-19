"""X-trainer real-robot deployment transport.

This package contains only the LAN transport contract used by X-trainer
deployment. It intentionally does not import policy model code.
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
