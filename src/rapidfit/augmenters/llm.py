"""LLM-based data augmenter."""

import random

import json_repair
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rapidfit.augmenters.base import BaseAugmenter
from rapidfit.clients import ChatClient
from rapidfit.io import DataSaver
from rapidfit.prompts import load_prompt
from rapidfit.types import AugmentResult, ClassInstruction, Sample, SaveFormat, SeedData, TaskPrompts, WriteMode


class LLMAugmenter(BaseAugmenter):
    """Augmenter that uses LLM to generate synthetic samples."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_id: str = "gpt-4.1-mini",
        max_samples_per_task: int = 128,
        batch_size: int = 8,
        max_temperature: float = 0.9,
        save_path: str = "./saved",
        save_format: SaveFormat | str = SaveFormat.JSON,
        save_incremental: bool = True,
        write_mode: WriteMode | str = WriteMode.OVERWRITE,
    ) -> None:
        """
        Initialize LLM augmenter.

        Args:
            api_key: OpenAI API key.
            base_url: Optional custom API base URL.
            model_id: Model to use for generation.
            max_samples_per_task: Maximum samples per task after augmentation.
            batch_size: Number of samples to generate per batch.
            max_temperature: Maximum temperature for random sampling.
            save_path: Directory to save generated data.
            save_format: Output format (json, jsonl, csv).
            save_incremental: Save while generating instead of at the end.
            write_mode: How to handle existing files (overwrite or append).
        """
        self._client = ChatClient(api_key, base_url, model_id)
        self._max_samples = max_samples_per_task
        self._batch_size = batch_size
        self._max_temp = max_temperature
        self._analyze_template = load_prompt("analyze")
        self._generate_template = load_prompt("generate")
        self._console = Console()
        self._saver = DataSaver(save_path, save_format)
        self._save_incremental = save_incremental
        self._write_mode = WriteMode(write_mode) if isinstance(write_mode, str) else write_mode

    def augment(self, seed_data: SeedData) -> AugmentResult:
        """
        Augment dataset using LLM generation.

        Args:
            seed_data: Mapping of task names to sample lists.

        Returns:
            Mapping of task names to results with file paths and stats.
        """
        task_prompts = self._analyze_tasks(seed_data)
        return self._generate_samples(seed_data, task_prompts)

    def _analyze_tasks(self, seed_data: SeedData) -> TaskPrompts:
        """Analyze seed data and generate class prompts for each task."""
        task_prompts: TaskPrompts = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analyzing[/] {task.description}"),
            TextColumn("[dim]calls:{task.fields[calls]} tokens:{task.fields[tokens]}[/]"),
            console=self._console,
        ) as progress:
            task = progress.add_task("", total=len(seed_data), calls=0, tokens=0)

            for task_name, samples in seed_data.items():
                progress.update(
                    task,
                    description=f"[cyan]{task_name}[/]",
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )
                samples_text = "\n".join(
                    f"- [{s['label']}] {s['text']}" for s in samples
                )
                prompt = self._analyze_template.format(
                    task_name=task_name,
                    samples=samples_text,
                )
                response = self._client.complete(prompt)
                task_prompts[task_name] = json_repair.loads(response)
                progress.update(
                    task,
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )
                progress.advance(task)

        return task_prompts

    def _generate_samples(
        self,
        seed_data: SeedData,
        task_prompts: TaskPrompts,
    ) -> AugmentResult:
        """Generate augmented samples for each task."""
        result: AugmentResult = {}
        augmented: SeedData = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/]"),
            TextColumn("[dim]calls:{task.fields[calls]} tokens:{task.fields[tokens]}[/]"),
            console=self._console,
        ) as progress:
            for task_name, samples in seed_data.items():
                existing = self._load_existing(task_name)
                existing_texts = {s["text"] for s in existing}
                base_samples = existing if existing else list(samples)

                class_instructions = task_prompts.get(task_name, {})
                labels = list(class_instructions.keys())

                if not labels:
                    augmented[task_name] = base_samples
                    path = self._saver.save_task(task_name, base_samples)
                    result[task_name] = {
                        "path": path,
                        "stats": self._compute_stats(base_samples),
                    }
                    continue

                counts = self._calculate_distribution(labels, len(base_samples))
                augmented[task_name] = base_samples
                total_samples = sum(counts.values())

                if total_samples <= 0:
                    path = self._saver.save_task(task_name, augmented[task_name])
                    result[task_name] = {
                        "path": path,
                        "stats": self._compute_stats(augmented[task_name]),
                    }
                    continue

                task_id = progress.add_task(
                    f"[bold]{task_name}[/]",
                    total=total_samples,
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )

                for label in labels:
                    needed = counts[label]
                    if needed <= 0:
                        continue

                    progress.update(
                        task_id,
                        description=f"[bold]{task_name}[/] → [yellow]{label}[/]",
                        calls=self._client.call_count,
                        tokens=self._client.total_tokens,
                    )
                    new_samples = self._generate_for_class(
                        task_name, label, class_instructions[label], needed, existing_texts, progress, task_id
                    )
                    augmented[task_name].extend(new_samples)

                    if self._save_incremental:
                        self._saver.save_task(task_name, augmented[task_name])

                path = self._saver.save_task(task_name, augmented[task_name])
                result[task_name] = {
                    "path": path,
                    "stats": self._compute_stats(augmented[task_name]),
                }

        self._console.print(
            f"[bold green]Done![/] calls: {self._client.call_count}, tokens: {self._client.total_tokens}"
        )
        return result

    def _load_existing(self, task_name: str) -> list[Sample]:
        """Load existing samples if in append mode."""
        if self._write_mode == WriteMode.APPEND:
            return self._saver.load_task(task_name)
        return []

    def _compute_stats(self, samples: list[Sample]) -> dict:
        """Compute label distribution stats."""
        labels: dict[str, int] = {}
        for s in samples:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
        return {"total": len(samples), "labels": labels}

    def _calculate_distribution(self, labels: list[str], existing_count: int = 0) -> dict[str, int]:
        """Calculate how many samples to generate per class for balance."""
        remaining = max(0, self._max_samples - existing_count)
        per_class = remaining // len(labels)
        return {label: per_class for label in labels}

    def _generate_for_class(
        self,
        task_name: str,
        label: str,
        instruction: ClassInstruction,
        count: int,
        existing_texts: set[str],
        progress: Progress,
        task_id: int,
    ) -> list[Sample]:
        """Generate samples for a specific class using batched iteration."""
        collected: set[str] = set()
        languages = instruction.get("languages", {})
        lang_str = ", ".join(f"{k} ({int(v*100)}%)" for k, v in languages.items())
        length = instruction.get("length", {})
        do_items = instruction.get("do", [])[:3]

        while len(collected) < count:
            batch_count = min(self._batch_size, count - len(collected))
            temp = random.uniform(0.5, self._max_temp)

            prompt = self._generate_template.format(
                task_name=task_name,
                label=label,
                languages=lang_str or "English",
                style=instruction.get("style", "")[:100],
                length_min=length.get("min", 10),
                length_max=length.get("max", 500),
                do_list="\n".join(f"- {d}" for d in do_items) if do_items else "- Be creative and natural",
                edge_cases="Vary tone, context, and vocabulary across samples.",
                count=batch_count,
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
                        if len(collected) >= count:
                            break

        return [{"text": t, "label": label} for t in collected]
