"""Type definitions for RapidFit."""

from enum import Enum
from typing import TypedDict


class SaveFormat(Enum):
    """Supported output formats for saving data."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


class WriteMode(Enum):
    """Mode for handling existing data files."""
    OVERWRITE = "overwrite"
    APPEND = "append"


class Sample(TypedDict):
    """A single classification sample."""
    text: str
    label: str


class LengthStats(TypedDict):
    """Length statistics for samples."""
    min: int
    max: int
    avg: int


class EdgeCases(TypedDict):
    """Edge case generation instructions."""
    short: str
    long: str
    noisy: str


class ClassInstruction(TypedDict):
    """Generation instructions for a class."""
    languages: dict[str, float]
    style: str
    length: LengthStats
    do: list[str]
    avoid: list[str]
    patterns: str
    edge_cases: EdgeCases


class TaskStats(TypedDict):
    """Statistics for a generated task."""
    total: int
    labels: dict[str, int]


class TaskResult(TypedDict):
    """Result for a single task augmentation."""
    path: str
    stats: TaskStats


class Prediction(TypedDict):
    """A single prediction result."""
    label: str
    confidence: float


class TrainConfig(TypedDict, total=False):
    """Configuration for classifier training."""
    model_name: str
    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    dropout_rate: float
    label_smoothing: float
    use_mean_pooling: bool
    freeze_epochs: int
    patience: int
    test_size: float
    val_size: float
    use_class_weights: bool
    output_dir: str
    save_path: str


SeedData = dict[str, list[Sample]]
"""Mapping of task names to lists of samples."""

AugmentResult = dict[str, TaskResult]
"""Mapping of task names to augmentation results."""

TaskPrompts = dict[str, dict[str, ClassInstruction]]
"""Mapping of task names to class instructions."""

ClassifierConfig = dict[str, any]
"""Configuration dictionary for classifiers."""
