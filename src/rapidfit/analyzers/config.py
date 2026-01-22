"""Configuration for dataset analysis and refinement."""

from dataclasses import dataclass

from rapidfit.types import SaveFormat


@dataclass
class AnalysisConfig:
    """Dataset analysis configuration."""

    imbalance_ratio: float = 0.1
    length_z_threshold: float = 3.0
    min_length: int = 1
    check_duplicates: bool = True


@dataclass
class RefinementConfig:
    """Dataset refinement configuration."""

    max_per_label: int | None = None
    max_label_ratio: float | None = None
    remove_short: bool = False
    remove_long: bool = False
    length_z_threshold: float = 3.0
    remove_empty: bool = True
    remove_duplicates: bool = True
    save_path: str | None = None
    save_format: SaveFormat = SaveFormat.JSONL
