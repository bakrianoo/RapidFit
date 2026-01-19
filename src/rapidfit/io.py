"""Data I/O utilities for saving and loading datasets."""

import csv
import json
from pathlib import Path

from rapidfit.types import Sample, SaveFormat


class DataSaver:
    """Handles saving and loading classification data in various formats."""

    def __init__(
        self,
        save_path: str = "./saved",
        save_format: SaveFormat | str = SaveFormat.JSON,
    ) -> None:
        self._path = Path(save_path)
        self._format = SaveFormat(save_format) if isinstance(save_format, str) else save_format
        self._path.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, list[Sample]]) -> dict[str, str]:
        """Save complete dataset, one file per task."""
        paths = {}
        for task, samples in data.items():
            paths[task] = self.save_task(task, samples)
        return paths

    def save_task(self, task: str, samples: list[Sample]) -> str:
        """Save samples for a single task."""
        path = self._task_path(task)
        self._write(path, samples)
        return str(path.resolve())

    def load_task(self, task: str) -> list[Sample]:
        """Load samples for a single task. Returns empty list if not found."""
        path = self._task_path(task)
        if not path.exists():
            return []
        return self._read(path)

    def _task_path(self, task: str) -> Path:
        """Get file path for a task."""
        safe_name = task.replace("/", "_").replace("\\", "_")
        return self._path / f"{safe_name}.{self._format.value}"

    def _write(self, path: Path, samples: list[Sample]) -> None:
        """Write samples to file."""
        if self._format == SaveFormat.JSON:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False, default=str)
        elif self._format == SaveFormat.JSONL:
            with open(path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
        elif self._format == SaveFormat.CSV:
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "label"])
                writer.writeheader()
                writer.writerows(samples)

    def load_all(self) -> dict[str, list[Sample]]:
        """Load all task files from the save directory."""
        data = {}
        ext = f".{self._format.value}"
        for path in self._path.glob(f"*{ext}"):
            task = path.stem
            data[task] = self._read(path)
        return data

    def _read(self, path: Path) -> list[Sample]:
        """Read samples from file."""
        if self._format == SaveFormat.JSON:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        elif self._format == SaveFormat.JSONL:
            samples = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples
        elif self._format == SaveFormat.CSV:
            with open(path, encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
