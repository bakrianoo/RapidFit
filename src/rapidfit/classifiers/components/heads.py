"""Task-specific classification heads."""

import torch.nn as nn

from rapidfit.classifiers.config import ActivationType, HeadConfig

ACTIVATIONS: dict[ActivationType, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def build_head(
    input_size: int, num_labels: int, config: HeadConfig
) -> nn.Sequential:
    """Build a classification head with configurable architecture."""
    hidden_size = int(input_size * config.hidden_multiplier)
    activation_cls = ACTIVATIONS[config.activation]

    layers: list[nn.Module] = []
    in_features = input_size

    for _ in range(config.hidden_layers):
        layers.extend([
            nn.Linear(in_features, hidden_size),
            activation_cls(),
            nn.Dropout(config.dropout),
        ])
        in_features = hidden_size

    layers.append(nn.Linear(in_features, num_labels))
    return nn.Sequential(*layers)


class TaskHeads(nn.ModuleDict):
    """Container for task-specific classification heads."""

    def __init__(
        self,
        hidden_size: int,
        task_num_labels: dict[str, int],
        config: HeadConfig,
    ) -> None:
        heads = {
            task: build_head(hidden_size, num_labels, config)
            for task, num_labels in task_num_labels.items()
        }
        super().__init__(heads)
        self.config = config
