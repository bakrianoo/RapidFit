# Error Analysis Guide

A practical guide to understanding where your classifier fails and how to fix it.

---

## The Core Idea

Training is done. Your model hits 92% accuracy. Great, right?

Not so fast. That 8% of errors might be hiding something nasty:
- Maybe the model confuses "angry" with "frustrated" 80% of the time
- Maybe high-confidence predictions are often wrong (overconfident model)
- Maybe certain phrases always trip it up

**Error analysis** answers one question: *Where exactly is my model failing?*

```
    After Training
          ↓
    ┌─────────────────────────┐
    │   classifier.analyze()  │
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────────────────────┐
    │  • Confusion matrix (what confuses what) │
    │  • Per-class metrics (weak spots)        │
    │  • Actual error samples (debug them)     │
    └─────────────────────────────────────────┘
```

---

## Quick Start

```python
from rapidfit import MultiheadClassifier

# Train your classifier
classifier = MultiheadClassifier()
classifier.train(data)

# Analyze the test set (created automatically during training)
result = classifier.analyze()

# See results in console
classifier.display(result)
```

That's it. You get confusion matrices, per-class metrics, and the actual samples that were misclassified.

### How Data Splits Work

When you call `train()`, RapidFit automatically splits your data:

```
Your Data (100%)
      ↓
┌─────────────────────────────────────┐
│  train (81%)  │  val (9%)  │  test (10%)  │
└─────────────────────────────────────┘
      ↓              ↓            ↓
   Training      Early Stop    Final Eval
```

These splits are stored internally and available for analysis:

```python
# After training, analyze any split
result = classifier.analyze(split="test")        # Default
result = classifier.analyze(split="validation")  # Check val set
result = classifier.analyze(split="train")       # Check for overfitting
```

**Note**: Splits are only available in the same session after `train()`. If you load a saved model, you'll need to provide your own data for analysis.

---

## Understanding the Output

### What `analyze()` Returns

```python
result = classifier.analyze()

# result is a dictionary with this structure:
{
    "tasks": {
        "sentiment": {
            "accuracy": 0.923,
            "labels": ["negative", "neutral", "positive"],
            "confusion_matrix": [[45, 2, 1], [3, 38, 4], [0, 3, 52]],
            "class_metrics": {
                "negative": {"precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 48},
                "neutral": {"precision": 0.88, "recall": 0.84, "f1": 0.86, "support": 45},
                "positive": {"precision": 0.91, "recall": 0.95, "f1": 0.93, "support": 55},
            },
            "errors": [
                {
                    "text": "It's fine I guess, nothing special",
                    "true_label": "neutral",
                    "predicted_label": "negative",
                    "confidence": 0.87,
                },
                # ... more errors
            ],
        },
        # ... other tasks
    }
}
```

### Reading the Confusion Matrix

Rows are true labels, columns are predictions.

```
              Predicted
              neg  neu  pos
True  neg     45    2    1     ← 45 correct, 2 called neutral, 1 called positive
      neu      3   38    4     ← 3 called negative, 38 correct, 4 called positive
      pos      0    3   52     ← 0 called negative, 3 called neutral, 52 correct
```

**Look for off-diagonal numbers.** High numbers there mean the model confuses those classes.

### Per-Class Metrics

| Metric | What it tells you |
|--------|-------------------|
| Precision | When the model predicts this class, how often is it right? |
| Recall | Of all samples that are this class, how many did the model catch? |
| F1 | Balance between precision and recall |
| Support | How many samples of this class exist |

**Low recall?** The model misses this class (false negatives).
**Low precision?** The model over-predicts this class (false positives).

---

## Use Cases

### Use Case 1: "Find the most confused classes"

**Problem**: You need to know which label pairs the model can't distinguish.

**Solution**: Look at the confusion matrix for high off-diagonal values.

```python
result = classifier.analyze()

for task, analysis in result["tasks"].items():
    matrix = analysis["confusion_matrix"]
    labels = analysis["labels"]
    
    # Find the biggest confusion
    max_confusion = 0
    confused_pair = None
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and matrix[i][j] > max_confusion:
                max_confusion = matrix[i][j]
                confused_pair = (true_label, pred_label)
    
    if confused_pair:
        print(f"{task}: '{confused_pair[0]}' often predicted as '{confused_pair[1]}' ({max_confusion} times)")
```

**What to do about it**: 
- Add more training examples that distinguish these classes
- Check if the labels themselves are ambiguous (maybe merge them?)

---

### Use Case 2: "See the actual mistakes"

**Problem**: Numbers are abstract. You want to see what the model actually got wrong.

**Solution**: Errors are sorted by confidence (highest first). High-confidence errors are the worst—the model was sure but wrong.

```python
result = classifier.analyze()

for task, analysis in result["tasks"].items():
    print(f"\n{task} - Top errors:")
    for err in analysis["errors"][:5]:
        print(f"  Text: {err['text'][:60]}...")
        print(f"  True: {err['true_label']} → Predicted: {err['predicted_label']} ({err['confidence']:.0%})")
        print()
```

**What to do about it**:
- High-confidence errors often reveal labeling mistakes in your data
- Patterns in error texts suggest what features confuse the model

---

### Use Case 3: "Analyze validation vs test set"

**Problem**: You want to check performance on different splits.

**Solution**: Use the `split` parameter.

```python
# Analyze test set (default)
test_result = classifier.analyze(split="test")

# Analyze validation set
val_result = classifier.analyze(split="validation")

# Compare accuracy
for task in test_result["tasks"]:
    test_acc = test_result["tasks"][task]["accuracy"]
    val_acc = val_result["tasks"][task]["accuracy"]
    print(f"{task}: val={val_acc:.2%}, test={test_acc:.2%}")
```

**What to look for**:
- Similar accuracy → Good, your model generalizes
- Val much better than test → Possible data leakage or test set is harder
- Test much better than val → Unusual, check your splits

---

### Use Case 4: "Focus on specific tasks"

**Problem**: You have many tasks but only care about one right now.

**Solution**: Use the `tasks` parameter.

```python
# Only analyze sentiment
result = classifier.analyze(tasks=["sentiment"])

# Analyze just two tasks
result = classifier.analyze(tasks=["sentiment", "intent"])
```

---

### Use Case 5: "Find weak classes to improve"

**Problem**: Overall accuracy is good, but some classes perform poorly.

**Solution**: Check per-class F1 scores.

```python
result = classifier.analyze()

for task, analysis in result["tasks"].items():
    print(f"\n{task} - Classes needing attention:")
    
    for label, metrics in analysis["class_metrics"].items():
        if metrics["f1"] < 0.8:  # Threshold for "needs work"
            print(f"  {label}: F1={metrics['f1']:.2%} (precision={metrics['precision']:.2%}, recall={metrics['recall']:.2%})")
```

**What to do about it**:
- Low support + low F1 → Need more training samples for this class
- High support + low F1 → Class definition might be ambiguous

---

### Use Case 6: "Pretty console output"

**Problem**: You want a quick visual summary without parsing the dictionary.

**Solution**: Use `display()` for formatted tables.

```python
result = classifier.analyze()

# Show everything with default 10 errors per task
classifier.display(result)

# Show more error examples
classifier.display(result, max_errors=25)
```

Output looks like:

```
Task: sentiment
Accuracy: 92.30%

         Per-Class Metrics
┌──────────┬───────────┬────────┬───────┬─────────┐
│ Label    │ Precision │ Recall │ F1    │ Support │
├──────────┼───────────┼────────┼───────┼─────────┤
│ negative │ 93.75%    │ 93.75% │ 93.75%│ 48      │
│ neutral  │ 88.37%    │ 84.44% │ 86.36%│ 45      │
│ positive │ 91.23%    │ 94.55% │ 92.86%│ 55      │
└──────────┴───────────┴────────┴───────┴─────────┘

         Confusion Matrix
...
```

---

### Use Case 7: "Export results for reporting"

**Problem**: You need to save the analysis for a report or share with teammates.

**Solution**: The result is a plain dictionary—serialize it however you want.

```python
import json

result = classifier.analyze()

# Save as JSON
with open("analysis_report.json", "w") as f:
    json.dump(result, f, indent=2)

# Or extract specific metrics for a CSV
import csv

with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["task", "label", "precision", "recall", "f1", "support"])
    
    for task, analysis in result["tasks"].items():
        for label, m in analysis["class_metrics"].items():
            writer.writerow([task, label, m["precision"], m["recall"], m["f1"], m["support"]])
```

---

## Common Patterns and What They Mean

### Pattern: Diagonal confusion matrix

```
     50   1   0
      0  48   2
      1   0  49
```

**Meaning**: Model distinguishes classes well. Minor errors are expected.

### Pattern: One row spread across columns

```
     50   1   0
     15  20  15    ← "neutral" scattered everywhere
      0   2  48
```

**Meaning**: This class (neutral) is hard to learn. Might be inherently ambiguous or needs more/better samples.

### Pattern: Two classes confuse each other

```
     30  18   2    ← negative often called neutral
      0  50   0
     12   3  35    ← positive often called negative
```

**Meaning**: Label definitions might overlap. Consider merging or adding distinguishing examples.

### Pattern: High confidence errors

```
Text: "This is absolutely terrible"
True: negative → Predicted: positive (94%)
```

**Meaning**: Either a labeling error in your data, or the model learned the wrong pattern. Investigate this sample.

---

## Best Practices

1. **Always analyze before deploying** → Training metrics can be misleading. Check where the model actually fails.

2. **Look at errors, not just numbers** → A 90% accuracy model might fail catastrophically on important edge cases.

3. **High-confidence errors first** → These reveal the worst problems: the model is sure but wrong.

4. **Compare val and test** → Big differences suggest something is off with your splits or data.

5. **Track metrics over time** → Save analysis results to compare across training runs.

6. **Check low-support classes** → Classes with few samples often have unstable metrics.

---

## Quick Reference

| I want to... | Code |
|--------------|------|
| Analyze test set | `classifier.analyze()` |
| Analyze validation set | `classifier.analyze(split="validation")` |
| Analyze specific tasks | `classifier.analyze(tasks=["sentiment"])` |
| Show formatted output | `classifier.display(result)` |
| Show more errors | `classifier.display(result, max_errors=25)` |
| Get accuracy | `result["tasks"]["sentiment"]["accuracy"]` |
| Get confusion matrix | `result["tasks"]["sentiment"]["confusion_matrix"]` |
| Get per-class metrics | `result["tasks"]["sentiment"]["class_metrics"]` |
| Get error samples | `result["tasks"]["sentiment"]["errors"]` |

---

## What's Next?

- Check [HOW_MULTIHEAD_CLASSIFIER.md](HOW_MULTIHEAD_CLASSIFIER.md) for training configuration
- Check [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for production deployment
