"""Dataset adapters providing a uniform interface across traffic corpora."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .flow_extractor import FlowExtractor
from .byte_aggregator import ByteExtractor
from .label_encoder import LabelEncoder
from .utils import get_logger


class BaseAdapter:
    """Base dataset adapter with a common load API."""

    def __init__(self, data_dir: str, logger=None):
        self.data_dir = data_dir
        self.logger = logger or get_logger()

    def load_data(self) -> Tuple[List[np.ndarray], List[str]]:
        raise NotImplementedError

    def get_metadata(self) -> Dict:
        raise NotImplementedError


class ISCXVPN2016Adapter(BaseAdapter):
    def load_data(self) -> Tuple[List[np.ndarray], List[str]]:
        # Placeholder: in practice iterate PCAP files under data_dir
        self.logger.info("Loading ISCXVPN2016 (placeholder implementation)")
        return [], []

    def get_metadata(self) -> Dict:
        return {"name": "ISCXVPN2016"}


class ISCXTor2016Adapter(BaseAdapter):
    def load_data(self) -> Tuple[List[np.ndarray], List[str]]:
        self.logger.info("Loading ISCXTor2016 (placeholder implementation)")
        return [], []

    def get_metadata(self) -> Dict:
        return {"name": "ISCXTor2016"}


class USTCTFC2016Adapter(BaseAdapter):
    def load_data(self) -> Tuple[List[np.ndarray], List[str]]:
        self.logger.info("Loading USTCTFC2016 (placeholder implementation)")
        return [], []

    def get_metadata(self) -> Dict:
        return {"name": "USTCTFC2016"}


__all__ = ["BaseAdapter", "ISCXVPN2016Adapter", "ISCXTor2016Adapter", "USTCTFC2016Adapter"]
