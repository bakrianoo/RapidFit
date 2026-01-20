"""LLM-based data refiner using error analysis."""

import random
from collections import defaultdict

import json_repair
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rapidfit.clients import ChatClient
from rapidfit.io import DataSaver
from rapidfit.prompts import load_prompt
from rapidfit.types import (
    AnalysisResult,
    AugmentResult,
    RefinementInstruction,
    Sample,
    SaveFormat,
    SeedData,
    TaskAnalysis,
    WriteMode,
)


class LLMRefiner:
    """Refiner that generates targeted samples based on error analysis."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_id: str = "gpt-4.1-mini",
        max_refined_per_class: int = 16,
        batch_size: int = 8,
        min_f1_threshold: float = 0.9,
        include_zero_support: bool = False,
        max_temperature: float = 0.9,
        save_path: str = "./saved",
        save_format: SaveFormat | str = SaveFormat.JSONL,
    ) -> None:
        """
        Initialize LLM refiner.

        Args:
            api_key: OpenAI API key.
            base_url: Optional custom API base URL.
            model_id: Model to use for generation.
            max_refined_per_class: Maximum refined samples per weak class.
            batch_size: Samples to generate per LLM call.
            min_f1_threshold: Refine classes with F1 below this threshold.
            include_zero_support: Include classes with no test samples.
            max_temperature: Maximum temperature for generation.
            save_path: Directory to save refined data.
            save_format: Output format (json, jsonl, csv).
        """
        self._client = ChatClient(api_key, base_url, model_id)
        self._max_refined = max_refined_per_class
        self._batch_size = batch_size
        self._f1_threshold = min_f1_threshold
        self._include_zero_support = include_zero_support
        self._max_temp = max_temperature
        self._analyze_template = load_prompt("refine_analyze")
        self._generate_template = load_prompt("refine_generate")
        self._console = Console()
        self._saver = DataSaver(save_path, save_format)

    def refine(
        self,
        analysis: AnalysisResult,
        existing_data: SeedData,
    ) -> AugmentResult:
        """
        Refine dataset based on error analysis.

        Args:
            analysis: Analysis result from classifier.analyze().
            existing_data: Current training data to augment.

        Returns:
            Mapping of task names to results with file paths and stats.
        """
        refinement_prompts = self._analyze_errors(analysis)
        return self._generate_refined(analysis, existing_data, refinement_prompts)

    def _analyze_errors(
        self,
        analysis: AnalysisResult,
    ) -> dict[str, dict[str, RefinementInstruction]]:
        """Analyze errors and generate refinement instructions per task."""
        task_instructions: dict[str, dict[str, RefinementInstruction]] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing errors[/] {task.description}"),
            TextColumn("[dim]calls:{task.fields[calls]} tokens:{task.fields[tokens]}[/]"),
            console=self._console,
        ) as progress:
            task = progress.add_task("", total=len(analysis["tasks"]), calls=0, tokens=0)

            for task_name, task_analysis in analysis["tasks"].items():
                weak_classes = self._get_weak_classes(task_analysis)
                if not weak_classes:
                    progress.advance(task)
                    continue

                progress.update(
                    task,
                    description=f"[cyan]{task_name}[/]",
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )

                confusion_pairs = self._extract_confusions(task_analysis, weak_classes)
                error_samples = self._format_errors(task_analysis["errors"][:10])

                prompt = self._analyze_template.format(
                    task_name=task_name,
                    weak_classes=", ".join(weak_classes),
                    confusion_pairs=confusion_pairs,
                    error_samples=error_samples,
                )

                response = self._client.complete(prompt)
                instructions = json_repair.loads(response)
                task_instructions[task_name] = {
                    k: v for k, v in instructions.items() if k in weak_classes
                }

                progress.update(
                    task,
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )
                progress.advance(task)

        return task_instructions

    def _generate_refined(
        self,
        analysis: AnalysisResult,
        existing_data: SeedData,
        instructions: dict[str, dict[str, RefinementInstruction]],
    ) -> AugmentResult:
        """Generate refined samples for weak classes."""
        result: AugmentResult = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/]"),
            TextColumn("[dim]calls:{task.fields[calls]} tokens:{task.fields[tokens]}[/]"),
            console=self._console,
        ) as progress:
            for task_name, class_instructions in instructions.items():
                if not class_instructions:
                    continue

                base_samples = list(existing_data.get(task_name, []))
                existing_texts = {s["text"] for s in base_samples}
                task_errors = analysis["tasks"][task_name]["errors"]

                total = len(class_instructions) * self._max_refined
                task_id = progress.add_task(
                    f"[bold]{task_name}[/]",
                    total=total,
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )

                for label, instr in class_instructions.items():
                    progress.update(
                        task_id,
                        description=f"[bold]{task_name}[/] → [yellow]{label}[/]",
                    )

                    class_errors = [e for e in task_errors if e["true_label"] == label]
                    new_samples = self._generate_for_class(
                        task_name, label, instr, class_errors,
                        existing_texts, progress, task_id
                    )
                    base_samples.extend(new_samples)

                random.shuffle(base_samples)
                path = self._saver.save_task(task_name, base_samples)
                result[task_name] = {
                    "path": path,
                    "stats": self._compute_stats(base_samples),
                }

        self._console.print(
            f"[bold green]Refinement done![/] calls: {self._client.call_count}, tokens: {self._client.total_tokens}"
        )
        return result

    def _generate_for_class(
        self,
        task_name: str,
        label: str,
        instr: RefinementInstruction,
        errors: list,
        existing_texts: set[str],
        progress: Progress,
        task_id: int,
    ) -> list[Sample]:
        """Generate refined samples for a single class."""
        collected: set[str] = set()
        languages = instr.get("languages", {})
        lang_str = ", ".join(f"{k} ({int(v*100)}%)" for k, v in languages.items())
        length = instr.get("length", {})

        error_examples = "\n".join(
            f"- \"{e['text'][:80]}\" (confused with {e['predicted_label']})"
            for e in errors[:5]
        ) or "No specific errors available"

        while len(collected) < self._max_refined:
            batch_count = min(self._batch_size, self._max_refined - len(collected))
            temp = random.uniform(0.5, self._max_temp)

            prompt = self._generate_template.format(
                task_name=task_name,
                label=label,
                count=batch_count,
                confused_with=", ".join(instr.get("confused_with", [])) or "other classes",
                differentiators="\n".join(f"- {d}" for d in instr.get("differentiators", [])) or "- Be distinct",
                emphasize="\n".join(f"- {e}" for e in instr.get("emphasize", [])) or "- Class-specific traits",
                avoid="\n".join(f"- {a}" for a in instr.get("avoid", [])) or "- Ambiguous patterns",
                languages=lang_str or "English",
                length_min=length.get("min", 10),
                length_max=length.get("max", 300),
                error_examples=error_examples,
            )

            response = self._client.complete(prompt, temperature=temp)
            texts = json_repair.loads(response)

            if isinstance(texts, list):
                for t in texts:
                    text = str(t).strip()
                    if text and text not in collected and text not in existing_texts:
                        collected.add(text)
                        existing_texts.add(text)
                        progress.update(
                            task_id,
                            calls=self._client.call_count,
                            tokens=self._client.total_tokens,
                        )
                        progress.advance(task_id)
                        if len(collected) >= self._max_refined:
                            break

        return [{"text": t, "label": label} for t in collected]

    def _get_weak_classes(self, task_analysis: TaskAnalysis) -> list[str]:
        """Get classes with F1 below threshold."""
        return [
            label for label, metrics in task_analysis["class_metrics"].items()
            if metrics["f1"] < self._f1_threshold
            and (metrics["support"] > 0 or self._include_zero_support)
        ]

    def _extract_confusions(
        self,
        task_analysis: TaskAnalysis,
        weak_classes: list[str],
    ) -> str:
        """Extract confusion pairs from errors."""
        confusions: dict[str, set[str]] = defaultdict(set)
        for error in task_analysis["errors"]:
            if error["true_label"] in weak_classes:
                confusions[error["true_label"]].add(error["predicted_label"])

        lines = []
        for true_label, pred_labels in confusions.items():
            lines.append(f"{true_label} -> {', '.join(pred_labels)}")
        return "\n".join(lines) or "No clear confusion patterns"

    def _format_errors(self, errors: list) -> str:
        """Format error samples for prompt."""
        if not errors:
            return "No errors available"
        return "\n".join(
            f"- \"{e['text'][:60]}...\" (true: {e['true_label']}, pred: {e['predicted_label']}, conf: {e['confidence']:.0%})"
            for e in errors
        )

    def _compute_stats(self, samples: list[Sample]) -> dict:
        """Compute label distribution stats."""
        labels: dict[str, int] = {}
        for s in samples:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
        return {"total": len(samples), "labels": labels}
