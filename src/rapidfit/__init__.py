"""RapidFit - Build multi-task classifiers and augment classification datasets."""

__version__ = "0.1.0"

from rapidfit.augmenters import BaseAugmenter, LLMAugmenter
from rapidfit.io import DataSaver
from rapidfit.types import AugmentResult, Sample, SaveFormat, SeedData, TaskResult, TaskStats

__all__ = [
    "AugmentResult",
    "BaseAugmenter",
    "DataSaver",
    "LLMAugmenter",
    "Sample",
    "SaveFormat",
    "SeedData",
    "TaskResult",
    "TaskStats",
]
