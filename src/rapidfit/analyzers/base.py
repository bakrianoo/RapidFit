"""Base analyzer interface."""

from abc import ABC, abstractmethod

from rapidfit.types import DatasetReport, SeedData


class BaseAnalyzer(ABC):
    """Abstract base class for dataset analyzers."""

    @abstractmethod
    def analyze(self, data: SeedData) -> DatasetReport:
        """Analyze dataset and return quality report."""
        pass
