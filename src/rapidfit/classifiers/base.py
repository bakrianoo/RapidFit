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
    def train(
        self,
        data: SeedData | AugmentResult | None = None,
        data_save_dir: str | None = None,
    ) -> None:
        """
        Train classifier on data.

        Args:
            data: Either SeedData dict or AugmentResult from augmenter.
            data_save_dir: Path to directory containing saved augmented data.
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

    def _resolve_data(
        self,
        data: SeedData | AugmentResult | None,
        data_save_dir: str | None,
    ) -> SeedData:
        """Resolve input to SeedData format."""
        if data_save_dir:
            return self._load_from_dir(data_save_dir)

        if not data:
            return {}

        first_value = next(iter(data.values()), None)
        if isinstance(first_value, list):
            return data

        return self._load_from_augment_result(data)

    def _load_from_dir(self, dir_path: str) -> SeedData:
        """Load all task data from a directory."""
        from rapidfit.io import DataSaver
        from rapidfit.types import SaveFormat

        path = Path(dir_path)
        if not path.exists():
            raise ValueError(f"Directory not found: {dir_path}")

        for fmt in SaveFormat:
            files = list(path.glob(f"*.{fmt.value}"))
            if files:
                return DataSaver(dir_path, fmt).load_all()
        return {}

    def _load_from_augment_result(self, data: AugmentResult) -> SeedData:
        """Load data from AugmentResult paths."""
        import csv
        import json

        resolved: SeedData = {}
        for task, result in data.items():
            file_path = Path(result.get("path", ""))
            if not file_path.exists():
                continue
            if file_path.suffix == ".jsonl":
                with open(file_path, encoding="utf-8") as f:
                    resolved[task] = [json.loads(line) for line in f if line.strip()]
            elif file_path.suffix == ".json":
                with open(file_path, encoding="utf-8") as f:
                    resolved[task] = json.load(f)
            elif file_path.suffix == ".csv":
                with open(file_path, encoding="utf-8") as f:
                    resolved[task] = list(csv.DictReader(f))
        return resolved
