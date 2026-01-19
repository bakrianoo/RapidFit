"""RapidFit - Build multi-task classifiers and augment classification datasets."""

__version__ = "0.1.0"

from rapidfit.augmenters import BaseAugmenter, LLMAugmenter
from rapidfit.classifiers import (
    BaseClassifier,
    ClassifierType,
    MultiheadClassifier,
    create_classifier,
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
    TrainConfig,
)

__all__ = [
    "AugmentResult",
    "BaseAugmenter",
    "BaseClassifier",
    "ClassifierConfig",
    "ClassifierType",
    "DataSaver",
    "LLMAugmenter",
    "MultiheadClassifier",
    "Prediction",
    "Sample",
    "SaveFormat",
    "SeedData",
    "TaskResult",
    "TaskStats",
    "TrainConfig",
    "create_classifier",
]
