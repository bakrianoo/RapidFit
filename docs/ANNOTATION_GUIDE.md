# Annotation Guide

A practical guide to labeling unlabeled data using LLM-powered annotation.

---

## The Core Idea

You have a pile of unlabeled texts. You know what labels you want. But manually labeling thousands of samples? That's brutal.

`LLMAnnotator` does the heavy lifting:

```
    Unlabeled Texts + Label Definitions
                    ↓
        ┌───────────────────────┐
        │    LLMAnnotator       │
        │  ─────────────────    │
        │  • Batch annotation   │
        │  • Label validation   │
        │  • Auto-save          │
        └───────────┬───────────┘
                    ↓
          Labeled Dataset (SeedData)
```

The result is ready to feed into `LLMAugmenter` or directly into `MultiheadClassifier`.

---

## Quick Start

```python
from rapidfit import LLMAnnotator

annotator = LLMAnnotator(
    api_key="your-api-key",
    save_path="./labeled",
)

texts = [
    "I absolutely love this product!",
    "Worst purchase I've ever made.",
    "It works, nothing special.",
    "Can you help me with my order?",
    # ... hundreds more
]

tasks = [
    {"name": "sentiment", "labels": ["positive", "negative", "neutral"]},
    {"name": "intent", "labels": ["feedback", "support", "question"]},
]

result = annotator.annotate(texts, tasks)
```

That's it. Each text gets labeled for each task. Results are saved incrementally so you don't lose progress if something crashes.

---

## Understanding the Output

### What `annotate()` Returns

```python
result = annotator.annotate(texts, tasks)

# result is a SeedData dictionary:
{
    "sentiment": [
        {"text": "I absolutely love this product!", "label": "positive"},
        {"text": "Worst purchase I've ever made.", "label": "negative"},
        {"text": "It works, nothing special.", "label": "neutral"},
    ],
    "intent": [
        {"text": "Can you help me with my order?", "label": "support"},
        # ...
    ],
}
```

### The Console Report

After annotation, you get a summary table per task:

```
┌─────────────────────────────────┐
│           sentiment             │
├──────────┬────────┬───────┬─────┤
│ Label    │ Labeled│ Synth │Total│
├──────────┼────────┼───────┼─────┤
│ positive │ 45     │ -     │ 45  │
│ negative │ 38     │ -     │ 38  │
│ neutral  │ 52     │ -     │ 52  │
├──────────┼────────┼───────┼─────┤
│ Total    │ 135    │ 0     │ 135 │
└──────────┴────────┴───────┴─────┘
```

- **Labeled**: Samples annotated from your input texts
- **Synth**: Samples synthesized for empty labels (when `fix_empty_labels=True`)
- **Total**: Combined count

---

## Use Cases

### Use Case 1: "Basic annotation with multiple tasks"

**Problem**: You have raw texts and want to label them for sentiment and intent simultaneously.

**Solution**: Define multiple tasks—LLM handles them in one pass.

```python
texts = load_your_texts()  # From file, database, wherever

tasks = [
    {"name": "sentiment", "labels": ["positive", "negative", "neutral"]},
    {"name": "intent", "labels": ["purchase", "support", "browse"]},
]

result = annotator.annotate(texts, tasks)
```

Each text is labeled for both tasks in a single LLM call per batch.

---

### Use Case 2: "Guide the LLM with hints"

**Problem**: Some labels are ambiguous. You want to steer the LLM's interpretation.

**Solution**: Add an `instruction` field to your task definition.

```python
tasks = [
    {
        "name": "urgency",
        "labels": ["urgent", "normal", "low"],
        "instruction": "Base urgency on time-sensitive keywords like 'asap', 'immediately', 'when you can'."
    },
]
```

The instruction is included in the prompt, helping the LLM understand your labeling criteria.

---

### Use Case 3: "Handle labels with no matching texts"

**Problem**: Your texts don't cover all labels. After annotation, some labels have zero samples.

**Solution**: Enable `fix_empty_labels` to synthesize samples for missing labels.

```python
annotator = LLMAnnotator(
    api_key="your-api-key",
    fix_empty_labels=True,
    min_samples_per_label=16,
)

result = annotator.annotate(texts, tasks)
```

**What happens**:
1. Annotator labels all texts normally
2. Checks which labels have zero samples
3. For each empty label, synthesizes samples using the labeled data as context
4. Report shows both labeled and synthesized counts

```
┌───────────────────────────────────┐
│            emotion                │
├──────────┬────────┬───────┬───────┤
│ Label    │ Labeled│ Synth │ Total │
├──────────┼────────┼───────┼───────┤
│ joy      │ 23     │ -     │ 23    │
│ anger    │ 18     │ -     │ 18    │
│ sadness  │ 0      │ 20    │ 20    │  ← synthesized
│ fear     │ 0      │ 20    │ 20    │  ← synthesized
├──────────┼────────┼───────┼───────┤
│ Total    │ 41     │ 40    │ 81    │
└──────────┴────────┴───────┴───────┘
```

---

### Use Case 4: "Control synthesis volume"

**Problem**: You want more (or fewer) synthesized samples per empty label.

**Solution**: Adjust `min_samples_per_label`.

```python
# Generate at least 32 samples per empty label
annotator = LLMAnnotator(
    api_key="your-api-key",
    fix_empty_labels=True,
    min_samples_per_label=32,
)
```

The actual count is `max(min_samples_per_label, average of non-empty labels)`. This keeps class balance reasonable.

---

### Use Case 5: "Resume interrupted annotation"

**Problem**: Annotation is expensive. If the process crashes halfway, you don't want to start over.

**Solution**: Just run again. With `overwrite=False` (default), existing data is loaded and already-annotated texts are skipped.

```python
annotator = LLMAnnotator(
    api_key="your-api-key",
    save_path="./labeled",
    overwrite=False,  # Default
)

# First run: processes all 1000 texts, crashes at 500
result = annotator.annotate(texts, tasks)

# Second run: loads 500 saved, processes remaining 500
result = annotator.annotate(texts, tasks)
```

The annotator checks which texts are already labeled and skips them. Synthesis also accounts for existing counts—if a label already has 20 samples and `min_samples_per_label=32`, it only synthesizes 12 more.

**To start fresh**: Set `overwrite=True` to ignore existing data.

---

### Use Case 6: "Balance underrepresented labels"

**Problem**: Some labels have samples, but not enough. You want to top them up to a minimum threshold.

**Solution**: Enable `augment_sparse_labels` to synthesize for labels below `min_samples_per_label`.

```python
annotator = LLMAnnotator(
    api_key="your-api-key",
    augment_sparse_labels=True,
    min_samples_per_label=32,
)

result = annotator.annotate(texts, tasks)
```

**What happens**:
1. Annotator labels all texts normally
2. Checks which labels have fewer than 32 samples
3. For each sparse label, synthesizes enough to reach 32
4. Report shows the breakdown

```
┌───────────────────────────────────┐
│            intent                 │
├──────────┬────────┬───────┬───────┤
│ Label    │ Labeled│ Synth │ Total │
├──────────┼────────┼───────┼───────┤
│ purchase │ 45     │ -     │ 45    │
│ support  │ 12     │ 20    │ 32    │  ← topped up
│ browse   │ 28     │ 4     │ 32    │  ← topped up
│ refund   │ 0      │ -     │ 0     │  ← not touched (use fix_empty_labels)
├──────────┼────────┼───────┼───────┤
│ Total    │ 85     │ 24    │ 109   │
└──────────┴────────┴───────┴───────┘
```

**Note**: This only affects labels with at least one sample. For labels with zero samples, use `fix_empty_labels`.

Combine both for full coverage:

```python
annotator = LLMAnnotator(
    api_key="your-api-key",
    fix_empty_labels=True,       # Handle zero-sample labels
    augment_sparse_labels=True,  # Top up low-count labels
    min_samples_per_label=32,
)
```

---

### Use Case 7: "Use a different LLM provider"

**Problem**: You want to use a local model or a different API provider.

**Solution**: Set `base_url` to point to any OpenAI-compatible endpoint.

```python
# Local Ollama
annotator = LLMAnnotator(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model_id="llama3",
)

# Azure OpenAI
annotator = LLMAnnotator(
    api_key=os.environ["AZURE_KEY"],
    base_url="https://your-resource.openai.azure.com/",
    model_id="gpt-4",
)
```

---

### Use Case 8: "Chain annotation into augmentation"

**Problem**: You want to annotate, then immediately augment the labeled data.

**Solution**: Pass the annotation result directly to `LLMAugmenter`.

```python
from rapidfit import LLMAnnotator, LLMAugmenter

# Step 1: Annotate
annotator = LLMAnnotator(api_key=key)
labeled = annotator.annotate(texts, tasks)

# Step 2: Augment
augmenter = LLMAugmenter(api_key=key, max_samples_per_task=256)
augmented = augmenter.augment(labeled)
```

The output of `annotate()` is `SeedData`, which is exactly what `augment()` expects.

---

### Use Case 9: "Train a classifier without augmentation"

**Problem**: You have enough annotated data and want to train directly.

**Solution**: Pass the annotation result straight to `MultiheadClassifier`.

```python
from rapidfit import LLMAnnotator, MultiheadClassifier

annotator = LLMAnnotator(api_key=key)
labeled = annotator.annotate(texts, tasks)

classifier = MultiheadClassifier()
classifier.train(labeled)
```

Or load from the saved directory:

```python
annotator = LLMAnnotator(api_key=key, save_path="./labeled")
annotator.annotate(texts, tasks)

classifier = MultiheadClassifier()
classifier.train(data_save_dir="./labeled")
```

No config files needed. Task names come from filenames, labels are inferred from samples.

---

## Configuration Reference

### LLMAnnotator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | required | OpenAI API key or compatible |
| `base_url` | `None` | Custom API endpoint |
| `model_id` | `"gpt-4.1-mini"` | Model to use |
| `batch_size` | `16` | Texts per LLM call |
| `temperature` | `0.3` | Lower = more consistent labels |
| `save_path` | `"./saved"` | Directory for output files |
| `save_format` | `"jsonl"` | Output format (json, jsonl, csv) |
| `save_incremental` | `True` | Save after each batch |
| `overwrite` | `False` | Ignore existing data and start fresh |
| `fix_empty_labels` | `False` | Synthesize for empty labels |
| `augment_sparse_labels` | `False` | Top up labels below minimum |
| `min_samples_per_label` | `16` | Target count for synthesis |

### Task Definition

```python
{
    "name": str,          # Required: task identifier
    "labels": list[str],  # Required: valid labels
    "instruction": str,   # Optional: guidance for LLM
}
```

---

## How It Works Internally

```
annotate(texts, tasks)
        │
        ▼
┌───────────────────────────────────────┐
│  1. Load existing data (if !overwrite) │
│  2. Filter out already-labeled texts   │
│  3. Batch remaining texts              │
│  4. For each batch:                    │
│     • Build prompt with task definitions│
│     • LLM returns JSON: {id: {task: label}} │
│     • Validate labels against task defs │
│     • Save valid samples                │
└───────────────────┬───────────────────┘
                    │
        if fix_empty_labels or
           augment_sparse_labels
                    │
                    ▼
┌───────────────────────────────────────┐
│  5. Count samples per label            │
│  6. For labels needing synthesis:      │
│     • Empty labels (if fix_empty)      │
│     • Sparse labels (if augment_sparse)│
│     • LLMSynthesizer generates samples │
│     • Uses labeled data as few-shot    │
└───────────────────┬───────────────────┘
                    │
                    ▼
           Final save + report
```

---

## Best Practices

1. **Start with a small batch** → Test on 50-100 texts first to verify label quality before running on thousands.

2. **Review a random sample** → LLMs make mistakes. Spot-check 5-10% of labels manually.

3. **Use low temperature** → Default 0.3 keeps labeling consistent. Higher values introduce randomness.

4. **Add instructions for edge cases** → If "neutral" keeps getting labeled as "positive", add a hint: "neutral means no clear sentiment, not mildly positive."

5. **Enable fix_empty_labels wisely** → Synthesized data is useful for bootstrapping, but real data is always better.

6. **Save incrementally** → API calls cost money. Don't lose progress to a network blip.

---

## Quick Reference

| I want to... | Code |
|--------------|------|
| Basic annotation | `annotator.annotate(texts, tasks)` |
| Add labeling hints | `{"name": "x", "labels": [...], "instruction": "hint"}` |
| Fix empty labels | `LLMAnnotator(..., fix_empty_labels=True)` |
| Top up sparse labels | `LLMAnnotator(..., augment_sparse_labels=True)` |
| Control synth count | `LLMAnnotator(..., min_samples_per_label=32)` |
| Resume annotation | Just run again (default behavior) |
| Start fresh | `LLMAnnotator(..., overwrite=True)` |
| Use local model | `LLMAnnotator(..., base_url="http://localhost:11434/v1")` |
| Save as JSON | `LLMAnnotator(..., save_format="json")` |
| Larger batches | `LLMAnnotator(..., batch_size=32)` |
| Chain to augmenter | `augmenter.augment(labeled)` |
| Train directly | `classifier.train(labeled)` |
| Train from saved dir | `classifier.train(data_save_dir="./labeled")` |

---

## What's Next?

- Check [REFINEMENT_GUIDE.md](REFINEMENT_GUIDE.md) for improving weak classes
- Check [HOW_MULTIHEAD_CLASSIFIER.md](HOW_MULTIHEAD_CLASSIFIER.md) for training
