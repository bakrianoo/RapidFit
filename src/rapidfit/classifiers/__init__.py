"""Classifier module exports and factory."""

from enum import Enum

from rapidfit.classifiers.base import BaseClassifier
from rapidfit.classifiers.multihead import MultiheadClassifier
from rapidfit.types import ClassifierConfig


class ClassifierType(Enum):
    """Available classifier types."""
    MULTIHEAD = "multihead"


def create_classifier(
    classifier_type: ClassifierType | str = ClassifierType.MULTIHEAD,
    config: ClassifierConfig | None = None,
) -> BaseClassifier:
    """
    Factory function to create classifiers.

    Args:
        classifier_type: Type of classifier to create.
        config: Optional configuration dict.

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
    "MultiheadClassifier",
    "create_classifier",
]
