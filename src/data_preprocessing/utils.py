"""Utility helpers for SD-MKD data preprocessing.

This module centralizes shared dataclasses and helper functions used across the
preprocessing pipeline, including five-tuple normalization, hashing utilities,
and lightweight logging setup.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FiveTuple:
    """A canonical five-tuple identifier for a network flow.

    Attributes
    ----------
    src_ip: str
        Source IPv4/IPv6 string.
    dst_ip: str
        Destination IPv4/IPv6 string.
    src_port: int
        Source transport-layer port.
    dst_port: int
        Destination transport-layer port.
    proto: str
        Transport protocol label (e.g., ``"TCP"`` or ``"UDP"``).
    """

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: str

    def canonical(self) -> "FiveTuple":
        """Return a direction-agnostic canonical ordering.

        The tuple is reordered such that the lexicographically smaller endpoint
        becomes ``src_*`` to merge bidirectional flows into a single key.
        """

        left: Tuple[str, int] = (self.src_ip, self.src_port)
        right: Tuple[str, int] = (self.dst_ip, self.dst_port)
        if left <= right:
            return self
        return FiveTuple(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            proto=self.proto,
        )

    def to_key(self) -> str:
        """Serialize the canonical tuple to a stable string key."""

        canon = self.canonical()
        return f"{canon.src_ip}:{canon.src_port}->{canon.dst_ip}:{canon.dst_port}:{canon.proto.upper()}"


@dataclass
class FlowPacket:
    """Minimal packet representation used throughout preprocessing."""

    timestamp: float
    payload: bytes
    five_tuple: FiveTuple


def five_tuple_from_tuple(values: Tuple[str, str, int, int, str]) -> FiveTuple:
    """Factory helper to build a :class:`FiveTuple` from a tuple."""

    src_ip, dst_ip, src_port, dst_port, proto = values
    return FiveTuple(src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port, proto=proto)


def ensure_bytes(payload: bytes | bytearray | memoryview | str) -> bytes:
    """Normalize various payload types to ``bytes``."""

    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, str):
        return payload.encode()
    raise TypeError(f"Unsupported payload type: {type(payload)}")


def compute_md5(text: str) -> str:
    """Compute an MD5 hex digest for a text key (used for cache names)."""

    return hashlib.md5(text.encode()).hexdigest()


def get_logger(name: str = "data_preprocessing", level: int = logging.INFO) -> logging.Logger:
    """Create or fetch a configured logger for preprocessing utilities."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = [
    "FiveTuple",
    "FlowPacket",
    "five_tuple_from_tuple",
    "ensure_bytes",
    "compute_md5",
    "get_logger",
]
