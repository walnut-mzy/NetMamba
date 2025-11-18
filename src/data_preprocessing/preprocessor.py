"""End-to-end preprocessing pipeline for SD-MKD traffic classification."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .adapters import ISCXVPN2016Adapter, ISCXTor2016Adapter, USTCTFC2016Adapter
from .byte_aggregator import ByteExtractor
from .cache_manager import CacheManager
from .data_splitter import generate_kfold_splits, stacking_split, train_val_test_split
from .dataset import FlowDataset
from .label_encoder import LabelEncoder
from .statistics import DatasetStatistics
from .validator import DataValidator
from .utils import get_logger


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing.

    Attributes mirror the detailed prompt to keep experiments reproducible.
    """

    dataset: str
    data_path: str
    packets_per_flow: int = 20
    bytes_per_packet: int = 150
    max_length: int = 1500
    image_height: int = 30
    image_width: int = 50
    batch_size: int = 512
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    cache: bool = True
    cache_dir: str = "./cache"


class DataPreprocessor:
    """Coordinate adapters, byte aggregation, validation, and splits."""

    def __init__(self, config: PreprocessConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self.cache_manager = CacheManager(cache_dir=config.cache_dir, logger=self.logger)
        self.validator = DataValidator(logger=self.logger)
        self.stats = DatasetStatistics(logger=self.logger)
        self.byte_extractor = ByteExtractor(
            packets_per_flow=config.packets_per_flow,
            bytes_per_packet=config.bytes_per_packet,
            max_length=config.max_length,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def process_dataset(self) -> Tuple[FlowDataset, LabelEncoder]:
        """Load raw data, aggregate bytes, validate, and optionally cache."""

        cache_key = json.dumps(asdict(self.config), sort_keys=True)
        if self.config.cache:
            cached = self.cache_manager.load(cache_key)
            if cached is not None:
                flows, labels, label_to_index = cached
                encoder = LabelEncoder(list(label_to_index.keys()))
                encoder.label_to_index = label_to_index
                encoder.index_to_label = {v: k for k, v in label_to_index.items()}
                return FlowDataset(flows, labels), encoder

        flows_raw, labels_text = self._load_raw()
        encoder = LabelEncoder(labels_text if labels_text else [])
        labels = encoder.encode(labels_text)

        flow_arrays: List[np.ndarray] = []
        for item in flows_raw:
            if isinstance(item, np.ndarray):
                arr = item
            elif hasattr(item, "packets"):
                arr = self.byte_extractor.get_flow_bytes(item.packets)  # type: ignore[arg-type]
            elif isinstance(item, (list, tuple)) and item and hasattr(item[0], "payload"):
                arr = self.byte_extractor.get_flow_bytes(item)  # type: ignore[arg-type]
            else:
                arr = np.array(item, dtype=np.uint8)
            flow_arrays.append(arr)

        num_classes = encoder.get_num_classes() if labels else 0
        self.validator.validate(flow_arrays, labels, num_classes)
        self.stats.summarize(flow_arrays, labels)

        dataset = FlowDataset(flow_arrays, labels, height=self.config.image_height, width=self.config.image_width)
        if self.config.cache:
            self.cache_manager.save(cache_key, (flow_arrays, labels, encoder.label_to_index))
        return dataset, encoder

    # ------------------------------------------------------------------
    # Loader helpers
    # ------------------------------------------------------------------
    def _load_raw(self) -> Tuple[List[object], List[str]]:
        adapter = self._get_adapter()
        if adapter:
            return adapter.load_data()
        self.logger.warning("No adapter matched; returning empty placeholder data")
        return [], []

    def _get_adapter(self):
        name = self.config.dataset.lower()
        if name == "iscxvpn2016":
            return ISCXVPN2016Adapter(self.config.data_path, logger=self.logger)
        if name == "iscxtor2016":
            return ISCXTor2016Adapter(self.config.data_path, logger=self.logger)
        if name == "ustctfc2016":
            return USTCTFC2016Adapter(self.config.data_path, logger=self.logger)
        return None

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------
    def split(self, dataset: FlowDataset):
        indices = list(range(len(dataset)))
        return train_val_test_split(indices, self.config.val_ratio, self.config.test_ratio, self.config.seed)

    def split_for_stacking(self, train_indices: Sequence[int]):
        return stacking_split(train_indices, stacking_ratio=0.3, seed=self.config.seed)

    def kfold(self, dataset: FlowDataset, labels: Sequence[int]):
        indices = list(range(len(dataset)))
        return list(generate_kfold_splits(indices, labels, stratified=True, seed=self.config.seed))


__all__ = ["PreprocessConfig", "DataPreprocessor"]
