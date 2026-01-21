"""LLM-based text annotator for unlabeled data."""

import json_repair
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from rapidfit.clients import ChatClient
from rapidfit.io import DataSaver
from rapidfit.prompts import load_prompt
from rapidfit.types import Sample, SaveFormat, SeedData, TaskDefinition
from rapidfit.augmenters.synthesizer import LLMSynthesizer


class LLMAnnotator:
    """Annotator that uses LLM to label unlabeled texts."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_id: str = "gpt-4.1-mini",
        batch_size: int = 16,
        temperature: float = 0.3,
        save_path: str = "./saved",
        save_format: SaveFormat | str = SaveFormat.JSONL,
        save_incremental: bool = True,
        fix_empty_labels: bool = False,
        min_samples_per_label: int = 16,
    ) -> None:
        self._client = ChatClient(api_key, base_url, model_id)
        self._batch_size = batch_size
        self._temperature = temperature
        self._template = load_prompt("annotate")
        self._console = Console()
        self._saver = DataSaver(save_path, save_format)
        self._save_incremental = save_incremental
        self._fix_empty = fix_empty_labels
        self._min_samples = min_samples_per_label
        self._synthesizer = LLMSynthesizer(api_key, base_url, model_id) if fix_empty_labels else None
        self._synth_counts: dict[str, dict[str, int]] = {}

    def annotate(
        self,
        texts: list[str],
        tasks: list[TaskDefinition],
    ) -> SeedData:
        """
        Annotate unlabeled texts with LLM.

        Args:
            texts: List of texts to annotate.
            tasks: List of task definitions with labels.

        Returns:
            Mapping of task names to annotated samples.
        """
        task_map = {t["name"]: set(t["labels"]) for t in tasks}
        result: SeedData = {t["name"]: [] for t in tasks}
        unique_texts: set[str] = set()

        task_defs = "\n".join(self._format_task(t) for t in tasks)
        task_schema = ", ".join(f'"{t["name"]}": "label"' for t in tasks)

        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Annotating[/] {task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/]"),
            TextColumn("[dim]calls:{task.fields[calls]} tokens:{task.fields[tokens]}[/]"),
            console=self._console,
        ) as progress:
            task_id = progress.add_task(
                "",
                total=len(texts),
                calls=self._client.call_count,
                tokens=self._client.total_tokens,
            )

            for batch in batches:
                progress.update(
                    task_id,
                    description=f"batch of {len(batch)}",
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )

                id_to_text = {str(i + 1): t for i, t in enumerate(batch)}
                texts_json = "\n".join(f'"{k}": "{v}"' for k, v in id_to_text.items())
                prompt = self._template.format(
                    task_definitions=task_defs,
                    texts=f"{{{texts_json}}}",
                    task_schema=task_schema,
                )

                response = self._client.complete(prompt, temperature=self._temperature)
                annotations = json_repair.loads(response)

                if isinstance(annotations, dict):
                    for text_id, labels in annotations.items():
                        text = id_to_text.get(str(text_id), "").strip()
                        if not text or text in unique_texts:
                            continue
                        if not isinstance(labels, dict):
                            continue

                        unique_texts.add(text)
                        for task_name, valid_labels in task_map.items():
                            label = str(labels.get(task_name, "")).strip()
                            if label in valid_labels:
                                result[task_name].append({"text": text, "label": label})

                progress.update(
                    task_id,
                    calls=self._client.call_count,
                    tokens=self._client.total_tokens,
                )
                progress.advance(task_id, len(batch))

                if self._save_incremental:
                    for task_name, samples in result.items():
                        if samples:
                            self._saver.save_task(task_name, samples)

        if self._fix_empty and self._synthesizer:
            self._fix_empty_labels(result, tasks)

        for task_name, samples in result.items():
            self._saver.save_task(task_name, samples)

        self._print_report(tasks, result)
        return result

    def _fix_empty_labels(self, result: SeedData, tasks: list[TaskDefinition]) -> None:
        """Synthesize samples for labels with no data."""
        for task in tasks:
            task_name = task["name"]
            samples = result.get(task_name, [])
            label_counts = {}
            for s in samples:
                label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

            empty_labels = [l for l in task["labels"] if label_counts.get(l, 0) == 0]
            if not empty_labels or not samples:
                continue

            non_empty = [c for c in label_counts.values() if c > 0]
            target = max(self._min_samples, sum(non_empty) // len(non_empty)) if non_empty else self._min_samples
            existing = {s["text"] for s in samples}
            self._synth_counts[task_name] = {}

            for label in empty_labels:
                new_samples = self._synthesizer.synthesize(
                    task_name, label, samples, target, existing
                )
                result[task_name].extend(new_samples)
                self._synth_counts[task_name][label] = len(new_samples)

    def _format_task(self, task: TaskDefinition) -> str:
        """Format task definition for prompt."""
        line = f"- {task['name']}: {', '.join(task['labels'])}"
        if task.get("instruction"):
            line += f"\n  Hint: {task['instruction']}"
        return line

    def _print_report(self, tasks: list[TaskDefinition], result: SeedData) -> None:
        """Print annotation summary report."""
        synth_calls = self._synthesizer.call_count if self._synthesizer else 0
        synth_tokens = self._synthesizer.total_tokens if self._synthesizer else 0
        self._console.print()
        self._console.print(
            f"[bold green]Done![/] calls: {self._client.call_count + synth_calls}, "
            f"tokens: {self._client.total_tokens + synth_tokens}"
        )
        self._console.print()

        for task in tasks:
            task_name = task["name"]
            samples = result.get(task_name, [])
            label_counts: dict[str, int] = {}
            for s in samples:
                label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

            synth_map = self._synth_counts.get(task_name, {})

            table = Table(title=f"[bold]{task_name}[/]", show_header=True)
            table.add_column("Label", style="cyan")
            table.add_column("Labeled", justify="right")
            table.add_column("Synth", justify="right")
            table.add_column("Total", justify="right")

            total_labeled = 0
            total_synth = 0
            for label in task["labels"]:
                total = label_counts.get(label, 0)
                synth = synth_map.get(label, 0)
                labeled = total - synth
                total_labeled += labeled
                total_synth += synth
                table.add_row(label, str(labeled), str(synth) if synth else "-", str(total))

            table.add_row("", "", "", "")
            table.add_row("[bold]Total[/]", f"[bold]{total_labeled}[/]", f"[bold]{total_synth}[/]", f"[bold]{len(samples)}[/]")

            self._console.print(table)
            self._console.print()
