"""Base analyzer interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from rapidfit.types import DatasetReport, SaveFormat, SeedData


class BaseAnalyzer(ABC):
    """Abstract base class for dataset analyzers."""

    @abstractmethod
    def analyze(
        self,
        data: SeedData | None = None,
        data_save_dir: str | None = None,
    ) -> DatasetReport:
        """Analyze dataset and return quality report."""
        pass

    def _load_from_dir(self, dir_path: str) -> SeedData:
        """Load all task data from a directory."""
        from rapidfit.io import DataSaver

        path = Path(dir_path)
        if not path.exists():
            raise ValueError(f"Directory not found: {dir_path}")

        for fmt in SaveFormat:
            files = list(path.glob(f"*.{fmt.value}"))
            if files:
                return DataSaver(dir_path, fmt).load_all()
        return {}
