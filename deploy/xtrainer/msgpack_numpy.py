"""MessagePack helpers for NumPy arrays used by the X-trainer transport."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import msgpack

    _IS_UMSGPACK = False
except ModuleNotFoundError:  # pragma: no cover - depends on the environment resolver
    import umsgpack as msgpack  # type: ignore[no-redef]

    _IS_UMSGPACK = True


PROTOCOL_VERSION = 1
SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_ARRAY_BYTES = 48 * 1024 * 1024
MAX_ARRAY_NDIM = 5
MAX_ARRAY_ELEMENTS = 64 * 1024 * 1024

_NDARRAY_MARKER = "__xtrainer_ndarray__"
_SUPPORTED_KINDS = {"b", "i", "u", "f"}


class ProtocolError(ValueError):
    """Raised when a MessagePack payload violates the X-trainer transport contract."""


def _validate_array(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind in {"O", "c"}:
        raise ProtocolError(f"unsupported array dtype: {array.dtype}")
    if array.dtype.kind not in _SUPPORTED_KINDS:
        raise ProtocolError(f"unsupported array dtype: {array.dtype}")
    if array.ndim > MAX_ARRAY_NDIM:
        raise ProtocolError(f"array has too many dimensions: {array.ndim}")
    if any(dim < 0 for dim in array.shape):
        raise ProtocolError(f"invalid array shape: {array.shape}")
    if array.size > MAX_ARRAY_ELEMENTS:
        raise ProtocolError(f"array has too many elements: {array.size}")
    if array.nbytes > MAX_ARRAY_BYTES:
        raise ProtocolError(f"array is too large: {array.nbytes} bytes")
    return np.ascontiguousarray(array)


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = _validate_array(value)
        return {
            _NDARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_NDARRAY_MARKER):
            return _decode_array(value)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(payload["dtype"])
        shape = tuple(int(dim) for dim in payload["shape"])
        data = payload["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid ndarray payload") from exc

    if not isinstance(data, (bytes, bytearray)):
        raise ProtocolError("ndarray data must be bytes")
    if dtype.kind in {"O", "c"} or dtype.kind not in _SUPPORTED_KINDS:
        raise ProtocolError(f"unsupported array dtype: {dtype}")
    if len(shape) > MAX_ARRAY_NDIM or any(dim < 0 for dim in shape):
        raise ProtocolError(f"invalid array shape: {shape}")

    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize if shape else dtype.itemsize
    if expected != len(data):
        raise ProtocolError(
            f"ndarray byte length mismatch: expected {expected} bytes, got {len(data)} bytes"
        )

    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return _validate_array(array).copy()


def dumps(payload: dict[str, Any], *, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> bytes:
    """Serialize a payload with NumPy array support."""

    try:
        encoded = _encode(payload)
        packed = msgpack.packb(encoded) if _IS_UMSGPACK else msgpack.packb(encoded, use_bin_type=True)
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError("payload contains unsupported objects") from exc
    if len(packed) > max_payload_bytes:
        raise ProtocolError(f"payload is too large: {len(packed)} bytes")
    return packed


def loads(buffer: bytes, *, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> dict[str, Any]:
    """Deserialize a payload and restore NumPy arrays."""

    if not isinstance(buffer, (bytes, bytearray)):
        raise ProtocolError("payload must be bytes")
    if len(buffer) > max_payload_bytes:
        raise ProtocolError(f"payload is too large: {len(buffer)} bytes")
    try:
        unpacked = msgpack.unpackb(buffer) if _IS_UMSGPACK else msgpack.unpackb(buffer, raw=False)
    except Exception as exc:
        raise ProtocolError("invalid MessagePack payload") from exc
    if not isinstance(unpacked, dict):
        raise ProtocolError("payload must be a map")
    return _decode(unpacked)


def protocol_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return transport metadata shared by health, handshake, and clients."""

    metadata = {
        "protocol": "xtrainer.websocket.msgpack",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "transport": "websocket",
        "encoding": "messagepack-numpy",
        "trusted_lan_only": True,
        "auth": "none",
        "tls": "none",
        "max_payload_bytes": MAX_PAYLOAD_BYTES,
    }
    if extra:
        metadata.update(extra)
    return metadata
