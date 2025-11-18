"""
Data loading scaffolding for encrypted traffic classification.

This module intentionally keeps preprocessing lightweight. In real deployments,
packet parsing and image construction would happen offline, producing an NPZ
archive (or any other serialized format) that stores tensors ready for
training. The dataset defined here only needs to yield a tensor shaped
``[1, H, W]`` and an ``int64`` label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetMeta:
    """Optional metadata to accompany an ``EncryptedFlowDataset`` instance."""

    split: str
    num_classes: int
    height: int
    width: int


class EncryptedFlowDataset(Dataset):
    """Dataset placeholder for encrypted flow images.

    Parameters
    ----------
    data_root:
        Directory that contains preprocessed NPZ files.
    split:
        A split identifier such as ``train_A``/``train_B``/``val_A``/``val_B``/``test``.
    transform:
        Optional callable applied to the image tensor (e.g., normalization or
        simple augmentations). The transform should accept and return a tensor
        shaped ``[1, H, W]``.

    Notes
    -----
    This implementation expects a file named ``{split}.npz`` under
    ``data_root`` that stores two arrays: ``images`` with shape
    ``[N, 1, H, W]`` and ``labels`` with shape ``[N]``.
    """

    def __init__(self, data_root: str, split: str, transform: Optional[Callable] = None):
        super().__init__()
        npz_path = f"{data_root}/{split}.npz"
        data = np.load(npz_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.images[idx], dtype=torch.float32)
        y = int(self.labels[idx])
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# Optional preprocessing recipe
# ---------------------------------------------------------------------------


def packets_to_tensor(packets, max_packets: int, bytes_per_packet: int, H: int, W: int):
    """Convert raw packets into a single-channel image tensor.

    Parameters
    ----------
    packets: list[bytes]
        Raw packet payloads ordered by capture time.
    max_packets: int
        Maximum packets to keep per flow. Shorter flows are padded with zeros.
    bytes_per_packet: int
        Number of bytes retained per packet. Each payload is truncated or padded
        with zeros to this length.
    H, W: int
        Target height and width of the output image. ``H * W`` must equal
        ``max_packets * bytes_per_packet``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[1, H, W]`` mapped to the range ``[0, 1]``.
    """

    # 1) Select the first ``max_packets`` packets; pad with empty packets if needed.
    packets = list(packets[:max_packets])
    if len(packets) < max_packets:
        packets.extend([b""] * (max_packets - len(packets)))

    # 2) Normalize each packet to ``bytes_per_packet`` bytes.
    normalized = []
    for p in packets:
        trimmed = p[:bytes_per_packet]
        if len(trimmed) < bytes_per_packet:
            trimmed = trimmed + b"\x00" * (bytes_per_packet - len(trimmed))
        normalized.append(trimmed)

    # 3) Concatenate all bytes and reshape.
    flat = np.frombuffer(b"".join(normalized), dtype=np.uint8).astype(np.float32)
    assert flat.size == max_packets * bytes_per_packet, "Unexpected flat size"
    tensor = flat.reshape(1, H, W) / 255.0  # map to [0,1]
    return torch.tensor(tensor, dtype=torch.float32)


__all__ = ["EncryptedFlowDataset", "DatasetMeta", "packets_to_tensor"]

