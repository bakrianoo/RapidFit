"""Classifier module exports and factory."""

from enum import Enum

from rapidfit.classifiers.base import BaseClassifier
from rapidfit.classifiers.config import (
    DEFAULT_CONFIG,
    EncoderConfig,
    EvalConfig,
    HeadConfig,
    LossConfig,
    MultiheadConfig,
    TrainingConfig,
)
from rapidfit.classifiers.export import export_to_onnx, quantize_onnx
from rapidfit.classifiers.multihead import MultiheadClassifier
from rapidfit.types import ClassifierConfig


class ClassifierType(Enum):
    """Available classifier types."""
    MULTIHEAD = "multihead"


def create_classifier(
    classifier_type: ClassifierType | str = ClassifierType.MULTIHEAD,
    config: ClassifierConfig | MultiheadConfig | None = None,
) -> BaseClassifier:
    """
    Factory function to create classifiers.

    Args:
        classifier_type: Type of classifier to create.
        config: Optional configuration dict or MultiheadConfig.

    Returns:
        Configured classifier instance.
    """
    if isinstance(classifier_type, str):
        classifier_type = ClassifierType(classifier_type)

    classifiers = {
        ClassifierType.MULTIHEAD: MultiheadClassifier,
    }

    cls = classifiers.get(classifier_type)
    if not cls:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return cls(config)


__all__ = [
    "BaseClassifier",
    "ClassifierType",
    "DEFAULT_CONFIG",
    "EncoderConfig",
    "EvalConfig",
    "HeadConfig",
    "LossConfig",
    "MultiheadClassifier",
    "MultiheadConfig",
    "TrainingConfig",
    "create_classifier",
    "export_to_onnx",
    "quantize_onnx",
]
