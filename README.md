# RapidFit

Turn a handful of labeled examples into a production-ready multi-task classifier.

RapidFit handles the two biggest pain points in text classification: **not enough data** and **too many separate models**. Give it a few examples per class, and it will generate more training data using LLMs, then train a single model that handles all your classification tasks at once.

## Installation

```bash
pip install rapidfit
```

## Annotate Unlabeled Data

Have raw texts but no labels? Use `LLMAnnotator` to get LLM-generated labels, then expand with augmentation.

```python
from rapidfit import LLMAnnotator, LLMAugmenter, MultiheadClassifier

# Define your tasks and labels
tasks = [
    {
        "name": "sentiment",
        "labels": ["positive", "negative", "neutral"],
        "instruction": "Judge by overall tone, not individual words"
    },
    {
        "name": "urgency",
        "labels": ["urgent", "normal", "low"]
    },
]

# Your unlabeled texts
texts = [
    "This product exceeded my expectations!",
    "Need immediate assistance with my order",
    "Just browsing, thanks.",
]

# Annotate with LLM
annotator = LLMAnnotator(api_key="your-api-key")
labeled_data = annotator.annotate(texts, tasks)

# Option 1: Expand with augmentation first
augmenter = LLMAugmenter(api_key="your-api-key", max_samples_per_task=128)
augmented = augmenter.augment(labeled_data)

# Option 2: Train directly on annotated data
classifier = MultiheadClassifier()
classifier.train(labeled_data)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `gpt-4.1-mini` | LLM to use for annotation |
| `batch_size` | `16` | Texts per annotation call |
| `temperature` | `0.3` | Lower for consistent labels |
| `save_path` | `./saved` | Output directory |
| `save_format` | `jsonl` | Format: `json`, `jsonl`, or `csv` |
| `fix_empty_labels` | `False` | Synthesize samples for labels with no data |
| `min_samples_per_label` | `16` | Minimum synthesized per empty label |

For the complete annotation workflow including label hints and synthesis options, see [Annotation Guide](docs/ANNOTATION_GUIDE.md).

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

## Error Analysis

Understand where your model fails before deploying:

```python
result = classifier.analyze()       # Analyze test set
classifier.display(result)          # Show confusion matrix, metrics, errors
```

Get per-class precision/recall, confusion matrices, and the actual samples that were misclassified. See [Error Analysis Guide](docs/ERROR_ANALYSIS_GUIDE.md) for details.

## Production Deployment

Export to ONNX for faster inference:

```bash
pip install rapidfit[export]
```

```python
# Export with INT8 quantization (3-4x faster, 4x smaller)
classifier.export_onnx("./onnx_models", quantize=True)
```

For deployment strategies, ONNX Runtime usage, and performance optimization, see [Inference Guide](docs/INFERENCE_GUIDE.md).

## Extend It

Build custom augmenters or classifiers by extending the base classes:

```python
from rapidfit import BaseAugmenter, BaseClassifier
```

## License

MIT
