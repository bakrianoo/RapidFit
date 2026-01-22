# Dataset Analysis Guide

A practical guide to detecting quality issues in your classification data before training.

---

## The Core Idea

Bad data → bad model. Before training, you want to catch:

- **Imbalanced labels** → Some classes dominate, others starve
- **Length outliers** → Suspiciously short or long texts
- **Empty texts** → Whitespace-only samples
- **Duplicates** → Same text appearing multiple times

`DatasetAnalyzer` scans your data and flags these issues. `DatasetRefiner` fixes them.

```
    SeedData (raw)
         ↓
┌────────────────────────┐
│   DatasetAnalyzer      │
│  ──────────────────    │
│  • Label distribution  │
│  • Length statistics   │
│  • Quality issues      │
└──────────┬─────────────┘
           ↓
    DatasetReport
           ↓
┌────────────────────────┐
│   DatasetRefiner       │
│  ──────────────────    │
│  • Cap per label       │
│  • Remove outliers     │
│  • Remove duplicates   │
└──────────┬─────────────┘
           ↓
    SeedData (clean)
```

---

## Quick Start

### Analyze

```python
from rapidfit import DatasetAnalyzer

analyzer = DatasetAnalyzer()

# From SeedData dict
report = analyzer.analyze(seed_data)

# Or from saved directory
report = analyzer.analyze(data_save_dir="./saved")
```

Output:

```
┌─────────────────────────────────────┐
│  sentiment (135 samples)            │
├───────────┬───────┬─────────────────┤
│ Label     │ Count │ Percent         │
├───────────┼───────┼─────────────────┤
│ positive  │ 89    │ 65.9%           │
│ negative  │ 42    │ 31.1%           │
│ neutral   │ 4     │ 3.0%            │
└───────────┴───────┴─────────────────┘
  Length: min=8 max=156 mean=42.3 std=28.1
  ● Under-represented labels: neutral
```

### Refine

```python
from rapidfit import DatasetRefiner, RefinementConfig

refiner = DatasetRefiner(RefinementConfig(
    max_per_label=50,
    remove_duplicates=True,
))

# From SeedData dict
cleaned = refiner.refine(seed_data)

# Or from saved directory
cleaned = refiner.refine(data_save_dir="./saved")
```

Output:

```
┌────────────────────────────────────┐
│        Refinement Summary          │
├─────────────┬──────────┬───────────┤
│ Task        │ Original │ Refined   │
├─────────────┼──────────┼───────────┤
│ sentiment   │ 135      │ 104       │
│ emotion     │ 89       │ 89        │
└─────────────┴──────────┴───────────┘
```

---

## Understanding the Report

### DatasetReport Structure

```python
report = analyzer.analyze(seed_data)

# report structure:
{
    "tasks": {
        "sentiment": {
            "total": 135,
            "labels": {
                "positive": {"count": 89, "percent": 65.9},
                "negative": {"count": 42, "percent": 31.1},
                "neutral": {"count": 4, "percent": 3.0},
            },
            "length": {"min": 8, "max": 156, "mean": 42.3, "std": 28.1},
            "issues": [
                {
                    "type": "imbalance",
                    "severity": "warning",
                    "message": "Under-represented labels: neutral",
                    "samples": [],
                },
            ],
        },
    },
    "total_issues": 1,
}
```

### Issue Types

| Type | Severity | Meaning |
|------|----------|---------|
| `imbalance` | warning | Label count below threshold relative to largest |
| `short_text` | warning | Unusually short texts (z-score outliers) |
| `long_text` | info | Unusually long texts (z-score outliers) |
| `empty` | error | Whitespace-only or zero-length texts |
| `duplicate` | warning | Same text appears multiple times |

---

## Use Cases

### Use Case 1: "Check data quality before training"

**Problem**: You want a quick sanity check before investing GPU hours.

**Solution**: Run analyzer and review the report.

```python
from rapidfit import DatasetAnalyzer

analyzer = DatasetAnalyzer()
report = analyzer.analyze(seed_data)

if report["total_issues"] > 0:
    print("Fix issues before training")
```

---

### Use Case 2: "Detect class imbalance"

**Problem**: Some labels have way more samples than others. This biases the model.

**Solution**: Analyzer flags labels below a threshold ratio.

```python
from rapidfit import DatasetAnalyzer, AnalysisConfig

analyzer = DatasetAnalyzer(AnalysisConfig(
    imbalance_ratio=0.2,  # Flag if label < 20% of largest
))
report = analyzer.analyze(seed_data)
```

Default threshold is 0.1 (10%). If your largest label has 100 samples, any label with fewer than 10 samples is flagged.

---

### Use Case 3: "Find length outliers"

**Problem**: Some samples are suspiciously short (typos?) or long (copy-paste errors?).

**Solution**: Analyzer uses z-score to detect statistical outliers.

```python
from rapidfit import DatasetAnalyzer, AnalysisConfig

analyzer = DatasetAnalyzer(AnalysisConfig(
    length_z_threshold=2.5,  # More sensitive (default: 3.0)
))
report = analyzer.analyze(seed_data)
```

A z-score of 3.0 means the text length is 3 standard deviations from the mean. The report shows the first few offending samples so you can inspect them.

---

### Use Case 4: "Cap samples per label"

**Problem**: One label has 500 samples, others have 50. You want balance.

**Solution**: Refiner caps each label to a maximum count.

```python
from rapidfit import DatasetRefiner, RefinementConfig

refiner = DatasetRefiner(RefinementConfig(
    max_per_label=100,
))
balanced = refiner.refine(seed_data)
```

Labels above 100 get trimmed. Labels below 100 stay untouched.

---

### Use Case 5: "Cap by ratio instead of count"

**Problem**: Different tasks have different scales. A fixed cap doesn't fit all.

**Solution**: Use `max_label_ratio` to cap relative to the largest label.

```python
from rapidfit import DatasetRefiner, RefinementConfig

refiner = DatasetRefiner(RefinementConfig(
    max_label_ratio=0.5,  # Each label capped at 50% of largest
))
balanced = refiner.refine(seed_data)
```

If your largest label has 200 samples, every label gets capped at 100.

---

### Use Case 6: "Remove length outliers"

**Problem**: Analyzer found short/long outliers. You want to remove them.

**Solution**: Enable outlier removal in refiner.

```python
from rapidfit import DatasetRefiner, RefinementConfig

refiner = DatasetRefiner(RefinementConfig(
    remove_short=True,
    remove_long=True,
    length_z_threshold=3.0,
))
cleaned = refiner.refine(seed_data)
```

Same z-score logic as analyzer. Samples beyond the threshold get removed.

---

### Use Case 7: "Remove empty and duplicate texts"

**Problem**: Data has garbage entries and copy-paste duplicates.

**Solution**: These are enabled by default.

```python
from rapidfit import DatasetRefiner, RefinementConfig

refiner = DatasetRefiner(RefinementConfig(
    remove_empty=True,      # Default
    remove_duplicates=True, # Default
))
cleaned = refiner.refine(seed_data)
```

First occurrence of a duplicate is kept. Subsequent ones are removed.

---

### Use Case 8: "Save refined data to disk"

**Problem**: You want to persist the cleaned dataset.

**Solution**: Set `save_path` in config.

```python
from rapidfit import DatasetRefiner, RefinementConfig, SaveFormat

refiner = DatasetRefiner(RefinementConfig(
    max_per_label=100,
    save_path="./refined",
    save_format=SaveFormat.JSONL,
))
result = refiner.refine(seed_data)  # Returns AugmentResult with paths
```

When `save_path` is set, `refine()` returns an `AugmentResult` (same format as augmenters) instead of `SeedData`. This lets you chain directly into training.

---

### Use Case 9: "Chain analysis → refinement → training"

**Problem**: Full pipeline from raw data to trained model.

**Solution**: Pipe outputs together.

```python
from rapidfit import (
    DatasetAnalyzer,
    DatasetRefiner,
    RefinementConfig,
    MultiheadClassifier,
)

# Analyze
analyzer = DatasetAnalyzer()
report = analyzer.analyze(seed_data)
print(f"Issues found: {report['total_issues']}")

# Refine
refiner = DatasetRefiner(RefinementConfig(
    max_per_label=100,
    remove_duplicates=True,
))
cleaned = refiner.refine(seed_data)

# Train
classifier = MultiheadClassifier()
classifier.train(cleaned)
```

---

### Use Case 10: "Load from saved directory"

**Problem**: Data is already saved from augmentation. You don't want to load it manually.

**Solution**: Pass `data_save_dir` instead of `data`.

```python
from rapidfit import DatasetAnalyzer, DatasetRefiner, RefinementConfig

# Analyze saved data
analyzer = DatasetAnalyzer()
report = analyzer.analyze(data_save_dir="./saved")

# Refine and save to new location
refiner = DatasetRefiner(RefinementConfig(
    max_per_label=100,
    save_path="./refined",
))
result = refiner.refine(data_save_dir="./saved")

# Train directly from refined directory
classifier = MultiheadClassifier()
classifier.train(data_save_dir="./refined")
```

---

### Use Case 11: "Remove corrupted or unwanted labels"

**Problem**: Some labels in your data are garbage (e.g., "unknown", "other", typos). You want to exclude them.

**Solution**: Use `ignore_labels` to specify labels to remove per task.

```python
from rapidfit import DatasetRefiner, RefinementConfig

refiner = DatasetRefiner(RefinementConfig(
    ignore_labels={
        "sentiment": ["unknown", "mixed"],
        "emotion": ["other", "n/a"],
    },
    save_path="./refined",
))
cleaned = refiner.refine(data_save_dir="./saved")
```

Samples with ignored labels are removed before any other refinement steps.

---

### Use Case 12: "Skip duplicate check for speed"

**Problem**: Duplicate detection is O(n). On large datasets, you want to skip it.

**Solution**: Disable in config.

```python
from rapidfit import DatasetAnalyzer, AnalysisConfig

analyzer = DatasetAnalyzer(AnalysisConfig(
    check_duplicates=False,
))
report = analyzer.analyze(large_dataset)
```

---

## Configuration Reference

### AnalysisConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imbalance_ratio` | `0.1` | Flag labels below this ratio of largest |
| `length_z_threshold` | `3.0` | Z-score threshold for length outliers |
| `min_length` | `1` | Minimum text length (below = empty) |
| `check_duplicates` | `True` | Check for duplicate texts |

### RefinementConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_per_label` | `None` | Maximum samples per label |
| `max_label_ratio` | `None` | Cap ratio relative to largest label |
| `ignore_labels` | `{}` | Labels to remove per task: `{"task": ["label1", ...]}` |
| `remove_short` | `False` | Remove short outliers |
| `remove_long` | `False` | Remove long outliers |
| `length_z_threshold` | `3.0` | Z-score threshold for removal |
| `remove_empty` | `True` | Remove whitespace-only texts |
| `remove_duplicates` | `True` | Remove duplicate texts |
| `save_path` | `None` | Directory to save refined data |
| `save_format` | `jsonl` | Output format (json, jsonl, csv) |

---

## How It Works Internally

### Analyzer

```
analyze(seed_data)
        │
        ▼
┌────────────────────────────────────┐
│  For each task:                    │
│  1. Count labels → distribution    │
│  2. Compute length stats → μ, σ    │
│  3. Check imbalance threshold      │
│  4. Check length z-scores          │
│  5. Check empty texts              │
│  6. Check duplicates (if enabled)  │
│  7. Collect issues                 │
└──────────────┬─────────────────────┘
               │
               ▼
        Print report
               │
               ▼
        Return DatasetReport
```

### Refiner

```
refine(seed_data)
        │
        ▼
┌────────────────────────────────────┐
│  For each task:                    │
│  1. Remove empty (if enabled)      │
│  2. Remove duplicates (if enabled) │
│  3. Filter length outliers         │
│  4. Cap per label                  │
└──────────────┬─────────────────────┘
               │
               ▼
        Print summary
               │
     ┌─────────┴─────────┐
     │                   │
  save_path?          no save_path
     │                   │
     ▼                   ▼
  Save files       Return SeedData
     │
     ▼
  Return AugmentResult
```

---

## Best Practices

1. **Always analyze before training** → Five seconds of analysis saves hours of debugging.

2. **Fix imbalance before augmentation** → Augmenting an imbalanced dataset amplifies the problem.

3. **Check outlier samples manually** → Before removing, peek at what's flagged. Sometimes "outliers" are valid edge cases you want to keep.

4. **Use ratio-based capping for multi-task** → Different tasks have different scales. `max_label_ratio` adapts automatically.

5. **Save refined data** → Don't re-run refinement every time. Persist the cleaned version.

---

## Quick Reference

| I want to... | Code |
|--------------|------|
| Basic analysis | `DatasetAnalyzer().analyze(data)` |
| Analyze from directory | `analyzer.analyze(data_save_dir="./saved")` |
| Custom imbalance threshold | `AnalysisConfig(imbalance_ratio=0.2)` |
| Stricter length detection | `AnalysisConfig(length_z_threshold=2.0)` |
| Skip duplicate check | `AnalysisConfig(check_duplicates=False)` |
| Refine from directory | `refiner.refine(data_save_dir="./saved")` |
| Ignore specific labels | `RefinementConfig(ignore_labels={"task": ["bad"]})` |
| Cap samples per label | `RefinementConfig(max_per_label=100)` |
| Cap by ratio | `RefinementConfig(max_label_ratio=0.5)` |
| Remove short texts | `RefinementConfig(remove_short=True)` |
| Remove long texts | `RefinementConfig(remove_long=True)` |
| Save refined data | `RefinementConfig(save_path="./refined")` |
| Chain to training | `classifier.train(data_save_dir="./refined")` |

---

## What's Next?

- Check [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) for labeling unlabeled data
- Check [REFINEMENT_GUIDE.md](REFINEMENT_GUIDE.md) for LLM-based refinement of weak classes
- Check [HOW_MULTIHEAD_CLASSIFIER.md](HOW_MULTIHEAD_CLASSIFIER.md) for training
