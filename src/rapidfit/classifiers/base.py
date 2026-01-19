"""Base classifier interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rapidfit.types import AugmentResult, ClassifierConfig, Prediction, SeedData


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._config = config or {}
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def train(self, data: SeedData | AugmentResult) -> None:
        """
        Train classifier on data.

        Args:
            data: Either SeedData dict or AugmentResult from augmenter.
        """
        pass

    @abstractmethod
    def predict(self, texts: list[str], task: str) -> list[Prediction]:
        """
        Predict labels for texts on a specific task.

        Args:
            texts: List of input texts.
            task: Task name to use for prediction.

        Returns:
            List of predictions with labels and confidence.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
        pass

    def _resolve_data(self, data: SeedData | AugmentResult) -> SeedData:
        """Convert AugmentResult to SeedData by loading from file paths."""
        if not data:
            return {}

        first_value = next(iter(data.values()), None)
        if isinstance(first_value, list):
            return data

        import json
        resolved: SeedData = {}
        for task, result in data.items():
            path = result.get("path", "")
            if not path:
                continue
            file_path = Path(path)
            if file_path.suffix == ".jsonl":
                with open(file_path, encoding="utf-8") as f:
                    resolved[task] = [json.loads(line) for line in f if line.strip()]
            elif file_path.suffix == ".json":
                with open(file_path, encoding="utf-8") as f:
                    resolved[task] = json.load(f)
            elif file_path.suffix == ".csv":
                import csv
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    resolved[task] = [{"text": r["text"], "label": r["label"]} for r in reader]
        return resolved
