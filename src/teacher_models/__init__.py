"""Teacher ensemble package for SD-MKD."""
from .base_models import (
    DenseNet121Teacher,
    MobileNetV3LargeTeacher,
    ResNet50Teacher,
    TeacherConfig,
    adjust_first_conv,
    build_all_teachers,
)
from .mlp_fusion import MLPFusion, StackingModel
from .ensemble import TeacherEnsemble
from .stacking_trainer import (
    BaseModelTrainer,
    EpochMetrics,
    evaluate_classifier,
    generate_meta_features,
    stacking_split,
    train_mlp_fusion,
    train_stacking_ensemble,
)
from .teacher_model import StackingTeacherModel
from .compat import TrainResult, train_single_teacher, train_stacking_model
from .utils import compute_accuracy, freeze_module, softmax_probs


__all__ = [
    "DenseNet121Teacher",
    "MobileNetV3LargeTeacher",
    "ResNet50Teacher",
    "TeacherConfig",
    "adjust_first_conv",
    "build_all_teachers",
    "MLPFusion",
    "StackingModel",
    "TeacherEnsemble",
    "BaseModelTrainer",
    "EpochMetrics",
    "evaluate_classifier",
    "generate_meta_features",
    "stacking_split",
    "train_mlp_fusion",
    "train_stacking_ensemble",
    "StackingTeacherModel",
    "TrainResult",
    "train_single_teacher",
    "train_stacking_model",
    "compute_accuracy",
    "freeze_module",
    "softmax_probs",
]
