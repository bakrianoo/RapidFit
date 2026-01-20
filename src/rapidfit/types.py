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


class ErrorSample(TypedDict):
    """A misclassified sample with details."""
    text: str
    true_label: str
    predicted_label: str
    confidence: float


class ClassMetrics(TypedDict):
    """Per-class performance metrics."""
    precision: float
    recall: float
    f1: float
    support: int


class TaskAnalysis(TypedDict):
    """Analysis results for a single task."""
    accuracy: float
    class_metrics: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    labels: list[str]
    errors: list[ErrorSample]


class AnalysisResult(TypedDict):
    """Complete analysis across all tasks."""
    tasks: dict[str, TaskAnalysis]


class RefinementInstruction(TypedDict):
    """Refinement instructions for a weak class."""
    confused_with: list[str]
    differentiators: list[str]
    emphasize: list[str]
    avoid: list[str]
    languages: dict[str, float]
    length: LengthStats


SeedData = dict[str, list[Sample]]
"""Mapping of task names to lists of samples."""

AugmentResult = dict[str, TaskResult]
"""Mapping of task names to augmentation results."""

TaskPrompts = dict[str, dict[str, ClassInstruction]]
"""Mapping of task names to class instructions."""

ClassifierConfig = dict[str, any]
"""Configuration dictionary for classifiers."""
