"""Loss functions for classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rapidfit.classifiers.config import LossConfig


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class TaskLoss:
    """Manages loss computation for multi-task learning."""

    def __init__(self, config: LossConfig) -> None:
        self.config = config
        self.class_weights: dict[str, torch.Tensor] = {}

    def set_class_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.class_weights = weights

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        weight = self.class_weights.get(task)
        if weight is not None:
            weight = weight.to(logits.device)

        if self.config.use_focal_loss:
            loss_fn = FocalLoss(
                gamma=self.config.focal_gamma,
                weight=weight,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            loss_fn = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=self.config.label_smoothing,
            )

        loss = loss_fn(logits, labels)
        task_weight = self.config.task_weights.get(task, 1.0)
        return loss * task_weight

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.class_weights.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.class_weights = state
