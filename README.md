# RapidFit

Build multi-task classifiers and augment classification datasets with ease.

## Features

- **Multi-task Classification**: Train classifiers that handle multiple classification tasks simultaneously
- **Data Augmentation**: Expand and enhance your classification datasets using LLM-based generation
- **Flexible Saving**: Save data in JSON, JSONL, or CSV formats with incremental or batch saving

## Installation

```bash
pip install rapidfit
```

## Development Installation

```bash
pip install -e .
```

## Quick Start

### Data Augmentation

```python
from rapidfit import LLMAugmenter, SaveFormat

# Prepare seed data
seed_data = {
    "sentiment-analysis": [
        {"text": "I love this product!", "label": "positive"},
        {"text": "Terrible experience.", "label": "negative"},
        {"text": "It's okay, nothing special.", "label": "neutral"},
    ],
    "emotion-analysis": [
        {"text": "This makes me so happy!", "label": "joy"},
        {"text": "I can't believe they did this.", "label": "anger"},
        {"text": "I miss the old days.", "label": "sadness"},
    ],
}

# Initialize augmenter
augmenter = LLMAugmenter(
    api_key="your-openai-api-key",
    base_url=None,  # Optional: custom API endpoint
    model_id="gpt-4.1-mini",  # Optional: model to use
    max_samples_per_task=128,  # Optional: max samples per task
    batch_size=8,  # Optional: samples per generation batch
    max_temperature=0.9,  # Optional: max temperature for sampling
    save_path="./saved",  # Optional: output directory
    save_format=SaveFormat.JSON,  # Optional: json, jsonl, or csv
    save_separated=False,  # Optional: separate file per task
    save_incremental=True,  # Optional: save while generating
)

# Augment dataset
augmented_data = augmenter.augment(seed_data)
```

### Save Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_path` | `"./saved"` | Directory for output files |
| `save_format` | `json` | Output format: `json`, `jsonl`, `csv` |
| `save_separated` | `False` | Create separate file for each task |
| `save_incremental` | `True` | Save progressively during generation |

### Custom Augmenter

Extend `BaseAugmenter` to create custom augmentation strategies:

```python
from rapidfit import BaseAugmenter, SeedData

class MyAugmenter(BaseAugmenter):
    def augment(self, seed_data: SeedData) -> SeedData:
        # Your augmentation logic here
        return seed_data
```

## Running

### Install Dependencies

```bash
pip install -e .
```

### Run Augmentation

```bash
export OPENAI_API_KEY="your-api-key"
python examples/test_augmentation.py
```

Optional environment variables:
- `OPENAI_BASE_URL` - Custom API endpoint
- `OPENAI_MODEL_ID` - Model to use (default: `gpt-4.1-mini`)

Output saves to `./saved/` directory.

## License

MIT
