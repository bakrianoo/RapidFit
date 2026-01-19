"""Pooling strategies for encoder outputs."""

import torch
import torch.nn as nn

from rapidfit.classifiers.config import PoolingStrategy


class Pooler(nn.Module):
    """Configurable pooling layer for transformer outputs."""

    def __init__(self, strategy: PoolingStrategy = "mean") -> None:
        super().__init__()
        self.strategy = strategy

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.strategy == "cls":
            return hidden_states[:, 0]
        if self.strategy == "max":
            return self._max_pool(hidden_states, attention_mask)
        return self._mean_pool(hidden_states, attention_mask)

    def _mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        return summed / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _max_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states = hidden_states.masked_fill(~mask.bool(), float("-inf"))
        return hidden_states.max(dim=1).values
