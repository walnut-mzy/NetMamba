"""SD-MKD data preprocessing package."""
from .adapters import ISCXVPN2016Adapter, ISCXTor2016Adapter, USTCTFC2016Adapter
from .byte_aggregator import ByteExtractor
from .cache_manager import CacheManager
from .data_splitter import generate_kfold_splits, stacking_split, train_val_test_split
from .dataset import FlowDataset, DatasetMeta
from .flow_extractor import FlowExtractor
from .label_encoder import LabelEncoder, calculate_class_weights
from .preprocessor import DataPreprocessor, PreprocessConfig
from .statistics import DatasetStatistics
from .validator import DataValidator

# Backwards compatibility with earlier scaffolding
EncryptedFlowDataset = FlowDataset

__all__ = [
    "ByteExtractor",
    "CacheManager",
    "DataPreprocessor",
    "PreprocessConfig",
    "FlowExtractor",
    "FlowDataset",
    "EncryptedFlowDataset",
    "DatasetMeta",
    "LabelEncoder",
    "calculate_class_weights",
    "DatasetStatistics",
    "DataValidator",
    "train_val_test_split",
    "stacking_split",
    "generate_kfold_splits",
    "ISCXVPN2016Adapter",
    "ISCXTor2016Adapter",
    "USTCTFC2016Adapter",
]
