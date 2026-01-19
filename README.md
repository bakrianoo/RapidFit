# RapidFit

Turn a handful of labeled examples into a production-ready multi-task classifier.

RapidFit handles the two biggest pain points in text classification: **not enough data** and **too many separate models**. Give it a few examples per class, and it will generate more training data using LLMs, then train a single model that handles all your classification tasks at once.

## Installation

```bash
pip install rapidfit
```

## Augment Your Data

Start with just a few examples. RapidFit uses LLMs to expand your dataset while preserving label quality.

```python
from rapidfit import LLMAugmenter

seed_data = {
    "sentiment": [
        {"text": "I love this product!", "label": "positive"},
        {"text": "Terrible experience.", "label": "negative"},
    ],
    "emotion": [
        {"text": "This makes me so happy!", "label": "joy"},
        {"text": "I can't believe they did this.", "label": "anger"},
    ],
}

augmenter = LLMAugmenter(api_key="your-api-key")
augmented = augmenter.augment(seed_data)
```

Configure generation with optional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `gpt-4.1-mini` | LLM to use for generation |
| `max_samples_per_task` | `128` | Target samples per task |
| `batch_size` | `8` | Samples per generation call |
| `save_path` | `./saved` | Output directory |
| `save_format` | `json` | Format: `json`, `jsonl`, or `csv` |
| `write_mode` | `overwrite` | `overwrite` or `append` to existing data |

When using `append` mode, RapidFit loads existing data from the save path and skips duplicate texts during generation.

## Train a Classifier

One model, multiple tasks. The multihead architecture shares a single encoder across all your classification tasks, making it efficient and consistent.

```python
from rapidfit import MultiheadClassifier

classifier = MultiheadClassifier()
classifier.train(augmented)
classifier.save("./model")
```

Or train directly from a saved data directory:

```python
classifier = MultiheadClassifier()
classifier.train(data_save_dir="./saved")
classifier.save("./model")
```

Customize training:

```python
from rapidfit import MultiheadConfig, TrainingConfig, LossConfig

config = MultiheadConfig(
    training=TrainingConfig(epochs=10, learning_rate=2e-5),
    loss=LossConfig(use_class_weights=True),
)
classifier = MultiheadClassifier(config)
```

For a complete guide on configuration options and use cases, see [How the Multihead Classifier Works](docs/HOW_MULTIHEAD_CLASSIFIER.md).

## Predict

```python
classifier = MultiheadClassifier()
classifier.load("./model")

# Single task
classifier.predict(["Great product!"], task="sentiment")
# [{"label": "positive", "confidence": 0.95}]

# All tasks
classifier.predict_all_tasks(["Great product!"])
```

## Extend It

Build custom augmenters or classifiers by extending the base classes:

```python
from rapidfit import BaseAugmenter, BaseClassifier
```

## License

MIT
