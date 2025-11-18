"""Byte extraction and stride-based aggregation utilities."""
from __future__ import annotations

import numpy as np
import torch
from typing import Iterable, List, Sequence

from .utils import FlowPacket, ensure_bytes


class ByteExtractor:
    """Convert ordered packets into fixed-length byte arrays.

    The extractor implements the stride-based strategy described in the SD-MKD
    prompt: keep the first ``packets_per_flow`` packets, truncate each payload to
    ``bytes_per_packet``, and flatten/pad to ``max_length`` bytes. The resulting
    arrays can be reshaped into ``[1, H, W]`` tensors for model ingestion.
    """

    def __init__(self, packets_per_flow: int = 20, bytes_per_packet: int = 150, max_length: int = 1500):
        self.packets_per_flow = packets_per_flow
        self.bytes_per_packet = bytes_per_packet
        self.max_length = max_length

    def extract_payload(self, packet: FlowPacket) -> bytes:
        """Return the application payload bytes from a packet structure."""

        return ensure_bytes(packet.payload)

    def get_flow_bytes(self, packets: Sequence[FlowPacket]) -> np.ndarray:
        """Aggregate a flow's packets into a fixed-length byte array."""

        selected = list(packets[: self.packets_per_flow])
        buffers: List[bytes] = []
        for pkt in selected:
            payload = self.extract_payload(pkt)
            trimmed = payload[: self.bytes_per_packet]
            if len(trimmed) < self.bytes_per_packet:
                trimmed = trimmed + b"\x00" * (self.bytes_per_packet - len(trimmed))
            buffers.append(trimmed)

        flat = b"".join(buffers)
        flat_bytes = np.frombuffer(flat, dtype=np.uint8)
        if flat_bytes.size > self.max_length:
            flat_bytes = flat_bytes[: self.max_length]
        elif flat_bytes.size < self.max_length:
            pad = np.zeros(self.max_length - flat_bytes.size, dtype=np.uint8)
            flat_bytes = np.concatenate([flat_bytes, pad], axis=0)
        return flat_bytes

    def to_image_tensor(self, flow_bytes: np.ndarray, H: int, W: int) -> torch.Tensor:
        """Reshape byte array into ``[1, H, W]`` and map to ``[0, 1]``."""

        assert flow_bytes.size == self.max_length, "Unexpected byte array length"
        assert H * W == self.max_length, "H*W must equal max_length"
        img = flow_bytes.astype(np.float32).reshape(1, H, W) / 255.0
        return torch.tensor(img, dtype=torch.float32)


__all__ = ["ByteExtractor"]
