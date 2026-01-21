"""LLM-based sample synthesizer for targeted label generation."""

import random

import json_repair
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rapidfit.clients import ChatClient
from rapidfit.prompts import load_prompt
from rapidfit.types import Sample


class LLMSynthesizer:
    """Synthesizer that generates samples for a target label using few-shot context."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_id: str = "gpt-4.1-mini",
        batch_size: int = 8,
        temperature: float = 0.7,
    ) -> None:
        self._client = ChatClient(api_key, base_url, model_id)
        self._batch_size = batch_size
        self._temperature = temperature
        self._template = load_prompt("synthesize")
        self._console = Console()

    @property
    def call_count(self) -> int:
        return self._client.call_count

    @property
    def total_tokens(self) -> int:
        return self._client.total_tokens

    def synthesize(
        self,
        task_name: str,
        target_label: str,
        context_samples: list[Sample],
        count: int,
        existing_texts: set[str] | None = None,
    ) -> list[Sample]:
        """
        Synthesize samples for a target label.

        Args:
            task_name: Name of the classification task.
            target_label: Label to generate samples for.
            context_samples: Labeled examples for few-shot context.
            count: Number of samples to generate.
            existing_texts: Texts to avoid duplicating.

        Returns:
            List of synthesized samples.
        """
        if existing_texts is None:
            existing_texts = set()

        collected: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Synthesizing[/] {task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/]"),
            TextColumn("[dim]calls:{task.fields[calls]} tokens:{task.fields[tokens]}[/]"),
            console=self._console,
        ) as progress:
            task_id = progress.add_task(
                f"[yellow]{target_label}[/]",
                total=count,
                calls=self._client.call_count,
                tokens=self._client.total_tokens,
            )

            while len(collected) < count:
                batch_count = min(self._batch_size, count - len(collected))
                examples = self._format_examples(context_samples)
                prompt = self._template.format(
                    task_name=task_name,
                    target_label=target_label,
                    examples=examples,
                    count=batch_count,
                )

                response = self._client.complete(prompt, temperature=self._temperature)
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

        return [{"text": t, "label": target_label} for t in collected]

    def _format_examples(self, samples: list[Sample], max_examples: int = 8) -> str:
        """Format context samples for prompt."""
        selected = random.sample(samples, min(len(samples), max_examples))
        return "\n".join(f"- [{s['label']}] {s['text']}" for s in selected)
