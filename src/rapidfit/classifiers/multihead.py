"""Multihead classifier implementation."""

import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from rapidfit.classifiers.base import BaseClassifier
from rapidfit.types import AugmentResult, Prediction, SeedData, TrainConfig

console = Console()

DEFAULT_CONFIG: TrainConfig = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout_rate": 0.2,
    "label_smoothing": 0.1,
    "use_mean_pooling": True,
    "freeze_epochs": 3,
    "patience": 3,
    "test_size": 0.1,
    "val_size": 0.1,
    "use_class_weights": True,
    "output_dir": "./training_output",
    "save_path": "./model",
}


class _MultiTaskModel(nn.Module):
    """Neural network with shared encoder and task-specific heads."""

    def __init__(
        self,
        model_name: str,
        task_num_labels: dict[str, int],
        task_label2id: dict[str, dict[str, int]],
        task_id2label: dict[str, dict[int, str]],
        dropout_rate: float = 0.2,
        use_mean_pooling: bool = True,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.task_num_labels = task_num_labels
        self.task_label2id = task_label2id
        self.task_id2label = task_id2label
        self.use_mean_pooling = use_mean_pooling
        self.label_smoothing = label_smoothing
        self.task_class_weights: dict[str, torch.Tensor] = {}

        hidden_size = self.encoder.config.hidden_size
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_labels),
            )
            for task, num_labels in task_num_labels.items()
        })
        self.dropout = nn.Dropout(dropout_rate)

    def set_class_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.task_class_weights = weights

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, num_layers: int | None = None) -> None:
        if num_layers is None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            return

        for param in self.encoder.parameters():
            param.requires_grad = False

        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
            for i in range(max(0, len(layers) - num_layers), len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        return summed / torch.clamp(mask.sum(dim=1), min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_name: str | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_mean_pooling:
            pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)
        logits = self.task_heads[task_name](pooled)

        loss = None
        if labels is not None:
            weight = self.task_class_weights.get(task_name)
            if weight is not None:
                weight = weight.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, "encoder"))

        weights_cpu = {k: v.cpu() for k, v in self.task_class_weights.items()}
        torch.save(
            {
                "task_heads": self.task_heads.state_dict(),
                "task_num_labels": self.task_num_labels,
                "task_label2id": self.task_label2id,
                "task_id2label": self.task_id2label,
                "use_mean_pooling": self.use_mean_pooling,
                "label_smoothing": self.label_smoothing,
                "task_class_weights": weights_cpu,
            },
            os.path.join(path, "task_heads.pt"),
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "_MultiTaskModel":
        config = torch.load(os.path.join(path, "task_heads.pt"), map_location="cpu")
        model = cls(
            model_name=os.path.join(path, "encoder"),
            task_num_labels=config["task_num_labels"],
            task_label2id=config["task_label2id"],
            task_id2label=config["task_id2label"],
            use_mean_pooling=config.get("use_mean_pooling", True),
            label_smoothing=config.get("label_smoothing", 0.1),
        )
        model.task_heads.load_state_dict(config["task_heads"])
        if "task_class_weights" in config:
            model.task_class_weights = config["task_class_weights"]
        return model


class _TaskGroupedSampler:
    """Sampler ensuring each batch contains samples from a single task."""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task_indices: dict[str, list[int]] = defaultdict(list)

        for idx in range(len(dataset)):
            self.task_indices[dataset[idx]["task"]].append(idx)

    def __iter__(self):
        batches = []
        for indices in self.task_indices.values():
            idx_copy = indices.copy()
            if self.shuffle:
                random.shuffle(idx_copy)
            for i in range(0, len(idx_copy), self.batch_size):
                batches.append(idx_copy[i : i + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(idx) + self.batch_size - 1) // self.batch_size
            for idx in self.task_indices.values()
        )


class _MultiTaskCollator:
    """Data collator handling task field and label validation."""

    def __init__(self, tokenizer: Any, task_num_labels: dict[str, int]) -> None:
        self.tokenizer = tokenizer
        self.task_num_labels = task_num_labels

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        tasks = [f.pop("task") for f in features]
        labels = [f.pop("label") for f in features]

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["task"] = tasks
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class _MultiTaskTrainer(Trainer):
    """Trainer with task-grouped batching."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        task_name = inputs.pop("task")[0]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_name=task_name,
            labels=labels,
        )
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def _get_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool):
        from torch.utils.data import DataLoader

        sampler = _TaskGroupedSampler(dataset, batch_size, shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_train_dataloader(self):
        return self._get_dataloader(
            self.train_dataset, self.args.per_device_train_batch_size, True
        )

    def get_eval_dataloader(self, eval_dataset=None):
        return self._get_dataloader(
            eval_dataset or self.eval_dataset,
            self.args.per_device_eval_batch_size,
            False,
        )


class MultiheadClassifier(BaseClassifier):
    """
    Classifier with shared encoder and task-specific heads.

    Each task gets its own classification head while sharing
    the underlying encoder for efficient multi-task learning.
    """

    def __init__(self, config: TrainConfig | None = None) -> None:
        super().__init__(config)
        self._cfg: TrainConfig = {**DEFAULT_CONFIG, **(config or {})}
        self._model: _MultiTaskModel | None = None
        self._tokenizer = None
        self._label_maps: dict[str, dict[str, int]] = {}
        self._id_maps: dict[str, dict[int, str]] = {}
        self._task_num_labels: dict[str, int] = {}

    def train(self, data: SeedData | AugmentResult) -> None:
        """Train multihead classifier on data."""
        samples = self._resolve_data(data)
        if not samples:
            raise ValueError("No training data provided")

        self._build_label_maps(samples)
        self._log_data_stats(samples)

        datasets, task_num_labels = self._prepare_datasets(samples)
        self._task_num_labels = task_num_labels

        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg["model_name"])
        self._model = _MultiTaskModel(
            model_name=self._cfg["model_name"],
            task_num_labels=task_num_labels,
            task_label2id=self._label_maps,
            task_id2label=self._id_maps,
            dropout_rate=self._cfg["dropout_rate"],
            use_mean_pooling=self._cfg["use_mean_pooling"],
            label_smoothing=self._cfg["label_smoothing"],
        )

        if self._cfg["use_class_weights"]:
            weights = self._compute_class_weights(datasets["train"], task_num_labels)
            self._model.set_class_weights(weights)

        tokenized = datasets.map(
            lambda x: self._tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=self._cfg["max_length"],
            ),
            batched=True,
            remove_columns=["text"],
        )

        collator = _MultiTaskCollator(self._tokenizer, task_num_labels)

        if self._cfg["freeze_epochs"] > 0:
            self._train_phase(
                tokenized,
                collator,
                epochs=self._cfg["freeze_epochs"],
                lr=1e-3,
                phase_name="heads-only",
                freeze_encoder=True,
            )

        self._train_phase(
            tokenized,
            collator,
            epochs=self._cfg["epochs"],
            lr=self._cfg["learning_rate"],
            phase_name="full",
            freeze_encoder=False,
        )

        self._is_trained = True
        console.print("[green]Training complete![/green]")

    def _train_phase(
        self,
        datasets: DatasetDict,
        collator: _MultiTaskCollator,
        epochs: int,
        lr: float,
        phase_name: str,
        freeze_encoder: bool,
    ) -> None:
        console.print(f"\n[bold cyan]Phase: {phase_name}[/bold cyan]")

        if freeze_encoder:
            self._model.freeze_encoder()
        else:
            self._model.unfreeze_encoder(num_layers=4)

        args = TrainingArguments(
            output_dir=os.path.join(self._cfg["output_dir"], phase_name),
            num_train_epochs=epochs,
            per_device_train_batch_size=self._cfg["batch_size"],
            per_device_eval_batch_size=self._cfg["batch_size"] * 2,
            learning_rate=lr,
            weight_decay=self._cfg["weight_decay"],
            warmup_ratio=self._cfg["warmup_ratio"],
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=50,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        callbacks = []
        if self._cfg["patience"] > 0:
            callbacks.append(EarlyStoppingCallback(self._cfg["patience"]))

        trainer = _MultiTaskTrainer(
            model=self._model,
            args=args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )

        trainer.train()

    @staticmethod
    def _compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    def predict(self, texts: list[str], task: str) -> list[Prediction]:
        """Predict labels for texts using task-specific head."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        if task not in self._model.task_heads:
            raise ValueError(f"Unknown task: {task}")

        self._model.eval()
        device = next(self._model.parameters()).device

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._cfg["max_length"],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task_name=task,
            )

        probs = torch.softmax(outputs["logits"], dim=-1)
        confidences, indices = probs.max(dim=-1)
        id2label = self._id_maps[task]

        return [
            {"label": id2label[idx.item()], "confidence": conf.item()}
            for idx, conf in zip(indices, confidences)
        ]

    def predict_all_tasks(self, texts: list[str]) -> dict[str, list[Prediction]]:
        """Predict all tasks for given texts."""
        return {task: self.predict(texts, task) for task in self._model.task_heads}

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        self._model.save_pretrained(str(path))
        self._tokenizer.save_pretrained(str(path / "encoder"))
        console.print(f"[green]Model saved to {path}[/green]")

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        self._model = _MultiTaskModel.from_pretrained(str(path))
        self._tokenizer = AutoTokenizer.from_pretrained(str(path / "encoder"))
        self._label_maps = self._model.task_label2id
        self._id_maps = self._model.task_id2label
        self._task_num_labels = self._model.task_num_labels
        self._is_trained = True
        console.print(f"[green]Model loaded from {path}[/green]")

    def _build_label_maps(self, data: SeedData) -> None:
        for task, samples in data.items():
            labels = sorted({s["label"] for s in samples})
            self._label_maps[task] = {lbl: i for i, lbl in enumerate(labels)}
            self._id_maps[task] = {i: lbl for i, lbl in enumerate(labels)}

    def _prepare_datasets(
        self, data: SeedData
    ) -> tuple[DatasetDict, dict[str, int]]:
        rows = []
        task_num_labels = {}

        for task, samples in data.items():
            label2id = self._label_maps[task]
            task_num_labels[task] = len(label2id)
            for s in samples:
                if s["label"] in label2id:
                    rows.append({
                        "text": s["text"],
                        "label": label2id[s["label"]],
                        "task": task,
                    })

        stratify_keys = [f"{r['task']}_{r['label']}" for r in rows]

        try:
            train, test = train_test_split(
                rows,
                test_size=self._cfg["test_size"],
                random_state=42,
                stratify=stratify_keys,
            )
            train_keys = [f"{r['task']}_{r['label']}" for r in train]
            train, val = train_test_split(
                train,
                test_size=self._cfg["val_size"],
                random_state=42,
                stratify=train_keys,
            )
        except ValueError:
            task_keys = [r["task"] for r in rows]
            train, test = train_test_split(
                rows, test_size=self._cfg["test_size"], random_state=42, stratify=task_keys
            )
            train_keys = [r["task"] for r in train]
            train, val = train_test_split(
                train, test_size=self._cfg["val_size"], random_state=42, stratify=train_keys
            )

        console.print(
            f"[dim]Splits: train={len(train)}, val={len(val)}, test={len(test)}[/dim]"
        )

        return (
            DatasetDict({
                "train": Dataset.from_list(train),
                "validation": Dataset.from_list(val),
                "test": Dataset.from_list(test),
            }),
            task_num_labels,
        )

    def _compute_class_weights(
        self, dataset: Dataset, task_num_labels: dict[str, int]
    ) -> dict[str, torch.Tensor]:
        weights = {}
        for task, num_labels in task_num_labels.items():
            counts = Counter(
                item["label"] for item in dataset if item["task"] == task
            )
            if not counts:
                continue

            total = sum(counts.values())
            w = torch.zeros(num_labels)
            for label_id in range(num_labels):
                count = counts.get(label_id, 1)
                w[label_id] = total / (num_labels * count)

            w = w / w.mean()
            weights[task] = torch.clamp(w, min=0.5, max=3.0)

        return weights

    def _log_data_stats(self, data: SeedData) -> None:
        table = Table(title="Dataset Statistics")
        table.add_column("Task", style="cyan")
        table.add_column("Samples", justify="right")
        table.add_column("Labels", justify="right")

        for task, samples in data.items():
            labels = {s["label"] for s in samples}
            table.add_row(task, str(len(samples)), str(len(labels)))

        console.print(table)

    @property
    def tasks(self) -> list[str]:
        """Get list of available tasks."""
        return list(self._label_maps.keys())

    def get_labels(self, task: str) -> list[str]:
        """Get label names for a task."""
        if task not in self._label_maps:
            raise ValueError(f"Unknown task: {task}")
        return list(self._label_maps[task].keys())
