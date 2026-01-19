"""Multihead classifier implementation."""

from pathlib import Path
from typing import Any

from rapidfit.classifiers.base import BaseClassifier
from rapidfit.types import AugmentResult, ClassifierConfig, Prediction, SeedData


class MultiheadClassifier(BaseClassifier):
    """
    Classifier with shared encoder and task-specific heads.
    
    Each task gets its own classification head while sharing
    the underlying encoder for efficient multi-task learning.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__(config)
        self._encoder = None
        self._heads: dict[str, Any] = {}
        self._label_maps: dict[str, dict[str, int]] = {}

    def train(self, data: SeedData | AugmentResult) -> None:
        """Train multihead classifier on data."""
        samples = self._resolve_data(data)
        self._build_label_maps(samples)
        # TODO: implement training logic
        raise NotImplementedError("Training not implemented")

    def predict(self, texts: list[str], task: str) -> list[Prediction]:
        """Predict labels for texts using task-specific head."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        if task not in self._heads:
            raise ValueError(f"Unknown task: {task}")
        # TODO: implement prediction logic
        raise NotImplementedError("Prediction not implemented")

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        # TODO: implement save logic
        raise NotImplementedError("Save not implemented")

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        # TODO: implement load logic
        raise NotImplementedError("Load not implemented")

    def _build_label_maps(self, data: SeedData) -> None:
        """Build label to index mappings for each task."""
        for task, samples in data.items():
            labels = sorted(set(s["label"] for s in samples))
            self._label_maps[task] = {label: i for i, label in enumerate(labels)}
