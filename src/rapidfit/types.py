"""Type definitions for RapidFit."""

from enum import Enum
from typing import TypedDict


class SaveFormat(Enum):
    """Supported output formats for saving data."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


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


SeedData = dict[str, list[Sample]]
"""Mapping of task names to lists of samples."""

AugmentResult = dict[str, TaskResult]
"""Mapping of task names to augmentation results."""

TaskPrompts = dict[str, dict[str, ClassInstruction]]
"""Mapping of task names to class instructions."""
