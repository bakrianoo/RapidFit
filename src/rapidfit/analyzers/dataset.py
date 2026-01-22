"""Dataset analysis and refinement."""

from collections import Counter

from rich.console import Console
from rich.table import Table

from rapidfit.analyzers.base import BaseAnalyzer
from rapidfit.analyzers.config import AnalysisConfig, RefinementConfig
from rapidfit.io import DataSaver
from rapidfit.types import (
    AugmentResult,
    DatasetIssue,
    DatasetReport,
    LabelDistribution,
    Sample,
    SeedData,
    TaskReport,
    TextLengthStats,
)


class DatasetAnalyzer(BaseAnalyzer):
    """Analyze dataset quality and detect issues."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self._config = config or AnalysisConfig()
        self._console = Console()

    def analyze(self, data: SeedData) -> DatasetReport:
        """Analyze dataset and print report."""
        tasks = {}
        total_issues = 0

        for task, samples in data.items():
            report = self._analyze_task(task, samples)
            tasks[task] = report
            total_issues += len(report["issues"])

        result: DatasetReport = {"tasks": tasks, "total_issues": total_issues}
        self._print_report(result)
        return result

    def _analyze_task(self, task: str, samples: list[Sample]) -> TaskReport:
        """Analyze a single task."""
        issues: list[DatasetIssue] = []
        labels = self._compute_label_dist(samples)
        length = self._compute_length_stats(samples)

        issues.extend(self._check_imbalance(labels))
        issues.extend(self._check_length_outliers(samples, length))
        issues.extend(self._check_empty(samples))

        if self._config.check_duplicates:
            issues.extend(self._check_duplicates(samples))

        return TaskReport(total=len(samples), labels=labels, length=length, issues=issues)

    def _compute_label_dist(self, samples: list[Sample]) -> dict[str, LabelDistribution]:
        """Compute label distribution."""
        counts = Counter(s["label"] for s in samples)
        total = len(samples) or 1
        return {
            label: LabelDistribution(count=c, percent=round(c / total * 100, 1))
            for label, c in counts.items()
        }

    def _compute_length_stats(self, samples: list[Sample]) -> TextLengthStats:
        """Compute text length statistics."""
        if not samples:
            return TextLengthStats(min=0, max=0, mean=0.0, std=0.0)

        lengths = [len(s["text"]) for s in samples]
        n = len(lengths)
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        std = variance ** 0.5

        return TextLengthStats(
            min=min(lengths),
            max=max(lengths),
            mean=round(mean, 1),
            std=round(std, 1),
        )

    def _check_imbalance(self, labels: dict[str, LabelDistribution]) -> list[DatasetIssue]:
        """Check for label imbalance."""
        if not labels:
            return []

        counts = [d["count"] for d in labels.values()]
        max_count = max(counts)
        threshold = max_count * self._config.imbalance_ratio

        weak = [l for l, d in labels.items() if d["count"] < threshold]
        if not weak:
            return []

        return [DatasetIssue(
            type="imbalance",
            severity="warning",
            message=f"Under-represented labels: {', '.join(weak)}",
            samples=[],
        )]

    def _check_length_outliers(
        self, samples: list[Sample], stats: TextLengthStats
    ) -> list[DatasetIssue]:
        """Check for length outliers."""
        if not samples or stats["std"] == 0:
            return []

        issues = []
        z = self._config.length_z_threshold
        short, long = [], []

        for s in samples:
            length = len(s["text"])
            z_score = (length - stats["mean"]) / stats["std"]
            if z_score < -z:
                short.append(s["text"][:50])
            elif z_score > z:
                long.append(s["text"][:50])

        if short:
            issues.append(DatasetIssue(
                type="short_text",
                severity="warning",
                message=f"{len(short)} unusually short texts",
                samples=short[:5],
            ))
        if long:
            issues.append(DatasetIssue(
                type="long_text",
                severity="info",
                message=f"{len(long)} unusually long texts",
                samples=long[:5],
            ))
        return issues

    def _check_empty(self, samples: list[Sample]) -> list[DatasetIssue]:
        """Check for empty or whitespace-only texts."""
        empty = [s["text"] for s in samples if not s["text"].strip()]
        if not empty:
            return []
        return [DatasetIssue(
            type="empty",
            severity="error",
            message=f"{len(empty)} empty texts",
            samples=empty[:5],
        )]

    def _check_duplicates(self, samples: list[Sample]) -> list[DatasetIssue]:
        """Check for duplicate texts."""
        seen = {}
        duplicates = []
        for s in samples:
            text = s["text"]
            if text in seen:
                duplicates.append(text[:50])
            seen[text] = True

        if not duplicates:
            return []
        return [DatasetIssue(
            type="duplicate",
            severity="warning",
            message=f"{len(duplicates)} duplicate texts",
            samples=duplicates[:5],
        )]

    def _print_report(self, report: DatasetReport) -> None:
        """Print analysis report to console."""
        for task, tr in report["tasks"].items():
            table = Table(title=f"[bold]{task}[/] ({tr['total']} samples)")
            table.add_column("Label")
            table.add_column("Count", justify="right")
            table.add_column("Percent", justify="right")

            for label, dist in sorted(tr["labels"].items(), key=lambda x: -x[1]["count"]):
                table.add_row(label, str(dist["count"]), f"{dist['percent']}%")

            self._console.print(table)

            ls = tr["length"]
            self._console.print(
                f"  [dim]Length: min={ls['min']} max={ls['max']} "
                f"mean={ls['mean']} std={ls['std']}[/]"
            )

            for issue in tr["issues"]:
                color = {"error": "red", "warning": "yellow", "info": "blue"}[issue["severity"]]
                self._console.print(f"  [{color}]● {issue['message']}[/]")
                for s in issue["samples"][:3]:
                    self._console.print(f"    [dim]{s}...[/]")

        if report["total_issues"] == 0:
            self._console.print("[green]No issues detected[/]")
        else:
            self._console.print(f"[yellow]Total issues: {report['total_issues']}[/]")


class DatasetRefiner:
    """Refine dataset by removing problematic samples."""

    def __init__(self, config: RefinementConfig | None = None) -> None:
        self._config = config or RefinementConfig()
        self._console = Console()
        self._saver = (
            DataSaver(self._config.save_path, self._config.save_format)
            if self._config.save_path
            else None
        )

    def refine(self, data: SeedData) -> SeedData | AugmentResult:
        """Refine dataset and optionally save."""
        refined = {}
        stats = {}

        for task, samples in data.items():
            result, task_stats = self._refine_task(samples)
            refined[task] = result
            stats[task] = task_stats

        self._print_stats(stats)

        if self._saver:
            return self._save(refined)
        return refined

    def _refine_task(self, samples: list[Sample]) -> tuple[list[Sample], dict]:
        """Refine samples for a single task."""
        original = len(samples)
        result = list(samples)

        if self._config.remove_empty:
            result = [s for s in result if s["text"].strip()]

        if self._config.remove_duplicates:
            seen = set()
            unique = []
            for s in result:
                if s["text"] not in seen:
                    seen.add(s["text"])
                    unique.append(s)
            result = unique

        if self._config.remove_short or self._config.remove_long:
            result = self._filter_by_length(result)

        if self._config.max_per_label:
            result = self._cap_per_label(result, self._config.max_per_label)
        elif self._config.max_label_ratio:
            result = self._cap_by_ratio(result, self._config.max_label_ratio)

        return result, {"original": original, "refined": len(result)}

    def _filter_by_length(self, samples: list[Sample]) -> list[Sample]:
        """Remove length outliers based on z-score."""
        if not samples:
            return samples

        lengths = [len(s["text"]) for s in samples]
        mean = sum(lengths) / len(lengths)
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        std = variance ** 0.5

        if std == 0:
            return samples

        z = self._config.length_z_threshold
        result = []
        for s in samples:
            z_score = (len(s["text"]) - mean) / std
            if self._config.remove_short and z_score < -z:
                continue
            if self._config.remove_long and z_score > z:
                continue
            result.append(s)
        return result

    def _cap_per_label(self, samples: list[Sample], max_count: int) -> list[Sample]:
        """Cap samples per label to max count."""
        by_label: dict[str, list[Sample]] = {}
        for s in samples:
            by_label.setdefault(s["label"], []).append(s)

        result = []
        for label, items in by_label.items():
            result.extend(items[:max_count])
        return result

    def _cap_by_ratio(self, samples: list[Sample], ratio: float) -> list[Sample]:
        """Cap samples per label to ratio of largest label."""
        counts = Counter(s["label"] for s in samples)
        if not counts:
            return samples

        max_count = max(counts.values())
        cap = int(max_count * ratio)
        return self._cap_per_label(samples, cap)

    def _save(self, data: SeedData) -> AugmentResult:
        """Save refined data and return AugmentResult."""
        result = {}
        for task, samples in data.items():
            path = self._saver.save_task(task, samples)
            labels = Counter(s["label"] for s in samples)
            result[task] = {
                "path": path,
                "stats": {"total": len(samples), "labels": dict(labels)},
            }
        return result

    def _print_stats(self, stats: dict[str, dict]) -> None:
        """Print refinement statistics."""
        table = Table(title="Refinement Summary")
        table.add_column("Task")
        table.add_column("Original", justify="right")
        table.add_column("Refined", justify="right")
        table.add_column("Removed", justify="right")

        for task, s in stats.items():
            removed = s["original"] - s["refined"]
            table.add_row(task, str(s["original"]), str(s["refined"]), str(removed))

        self._console.print(table)
