"""Configuration definitions for classifiers."""

from dataclasses import dataclass, field
from typing import Literal

PoolingStrategy = Literal["mean", "cls", "max"]
ActivationType = Literal["gelu", "relu", "silu", "tanh"]
TaskSampling = Literal["proportional", "equal", "sqrt"]
EarlyStoppingMetric = Literal["loss", "accuracy", "f1"]


@dataclass
class HeadConfig:
    """Task head architecture configuration."""

    hidden_layers: int = 1
    hidden_multiplier: float = 1.0
    activation: ActivationType = "gelu"
    dropout: float = 0.2


@dataclass
class EncoderConfig:
    """Encoder fine-tuning configuration."""

    freeze_epochs: int = 3
    unfreeze_layers: int | None = 4
    lr_multiplier: float = 0.1


@dataclass
class LossConfig:
    """Loss function configuration."""

    label_smoothing: float = 0.1
    use_class_weights: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    task_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    test_size: float = 0.1
    val_size: float = 0.1
    output_dir: str = "./training_output"
    save_path: str = "./model"


@dataclass
class EvalConfig:
    """Evaluation and early stopping configuration."""

    patience: int = 3
    metric: EarlyStoppingMetric = "accuracy"
    min_delta: float = 0.001


@dataclass
class MultiheadConfig:
    """Complete configuration for MultiheadClassifier."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    pooling: PoolingStrategy = "mean"
    task_sampling: TaskSampling = "proportional"

    @classmethod
    def from_dict(cls, data: dict) -> "MultiheadConfig":
        """Create config from dictionary."""
        return cls(
            training=TrainingConfig(**data.get("training", {})),
            head=HeadConfig(**data.get("head", {})),
            encoder=EncoderConfig(**data.get("encoder", {})),
            loss=LossConfig(**data.get("loss", {})),
            eval=EvalConfig(**data.get("eval", {})),
            pooling=data.get("pooling", "mean"),
            task_sampling=data.get("task_sampling", "proportional"),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


DEFAULT_CONFIG = MultiheadConfig()
