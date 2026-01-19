"""RapidFit - Build multi-task classifiers and augment classification datasets."""

__version__ = "0.1.4"

from rapidfit.augmenters import BaseAugmenter, LLMAugmenter
from rapidfit.classifiers import (
    BaseClassifier,
    ClassifierType,
    EncoderConfig,
    EvalConfig,
    HeadConfig,
    LossConfig,
    MultiheadClassifier,
    MultiheadConfig,
    TrainingConfig,
    create_classifier,
    export_to_onnx,
    quantize_onnx,
)
from rapidfit.io import DataSaver
from rapidfit.types import (
    AugmentResult,
    ClassifierConfig,
    Prediction,
    Sample,
    SaveFormat,
    SeedData,
    TaskResult,
    TaskStats,
    WriteMode,
)

__all__ = [
    "AugmentResult",
    "BaseAugmenter",
    "BaseClassifier",
    "ClassifierConfig",
    "ClassifierType",
    "DataSaver",
    "EncoderConfig",
    "EvalConfig",
    "HeadConfig",
    "LLMAugmenter",
    "LossConfig",
    "MultiheadClassifier",
    "MultiheadConfig",
    "Prediction",
    "Sample",
    "SaveFormat",
    "SeedData",
    "TaskResult",
    "TaskStats",
    "TrainingConfig",
    "WriteMode",
    "create_classifier",
    "export_to_onnx",
    "quantize_onnx",
]
