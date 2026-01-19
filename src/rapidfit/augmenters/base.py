"""Base augmenter interface."""

from abc import ABC, abstractmethod

from rapidfit.types import AugmentResult, SeedData


class BaseAugmenter(ABC):
    """Abstract base class for data augmenters."""

    @abstractmethod
    def augment(self, seed_data: SeedData) -> AugmentResult:
        """
        Augment the seed dataset.

        Args:
            seed_data: Mapping of task names to sample lists.

        Returns:
            Mapping of task names to results with file paths and stats.
        """
        pass
