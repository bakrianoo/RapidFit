# Data Refinement Guide

A practical guide to improving your classifier by generating targeted training data.

---

## The Core Idea

Your model hits 85% accuracy. Error analysis reveals the problem:
- "promotional" gets confused with "news-sharing"
- "humor-meme" often predicted as "rant"
- "motivational" mistaken for "quote"

More random training data won't help. You need **targeted data** that teaches the model to distinguish these specific classes.

**Data refinement** answers one question: *How do I generate samples that fix my model's weaknesses?*

```
    Error Analysis
          ↓
    ┌─────────────────────────────────────┐
    │  Weak classes + confusion patterns  │
    └───────────┬─────────────────────────┘
                ↓
    ┌─────────────────────────────────────┐
    │  LLMRefiner.refine()                │
    │  • Analyzes what causes confusion   │
    │  • Generates distinguishing samples │
    └───────────┬─────────────────────────┘
                ↓
    ┌─────────────────────────────────────┐
    │  Retrain with enriched data         │
    └─────────────────────────────────────┘
```

---

## Quick Start

```python
from rapidfit import MultiheadClassifier, LLMRefiner, DataSaver, SaveFormat

# Train and analyze
classifier = MultiheadClassifier()
classifier.train(data)
result = classifier.analyze()

# See what needs work
classifier.display(result)

# Load existing data
saver = DataSaver("./saved", SaveFormat.JSONL)
existing_data = saver.load_all()

# Refine the weak spots
refiner = LLMRefiner(
    api_key=api_key,
    min_f1_threshold=0.85,      # Target classes below this F1
    max_refined_per_class=16,   # Samples to generate per weak class
    save_path="./saved_refined",        # Save refined data here
    save_format=SaveFormat.JSONL,
)

refined_data = refiner.refine(result, existing_data)

# Retrain with enriched data
classifier.train(refined_data)
```

That's it. The refiner analyzes your errors, figures out what distinguishes confused classes, and generates samples that emphasize those differences.

---

## How It Works

### Two-Phase Process

Just like the original augmenter, the refiner uses two LLM phases:

**Phase 1: Analyze Errors**
```
Input: Error samples, confusion patterns, weak class metrics
   ↓
LLM analyzes what causes confusion
   ↓
Output: Refinement instructions per class
  • What distinguishes this class from confused classes
  • Patterns to emphasize
  • Patterns to avoid
```

**Phase 2: Generate Targeted Samples**
```
Input: Refinement instructions + error examples
   ↓
LLM generates samples that:
  • Clearly exhibit the target class traits
  • Avoid patterns that cause confusion
  • Cover edge cases from actual errors
   ↓
Output: New training samples for weak classes
```

---

## Understanding the Parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `min_f1_threshold` | 0.9 | Only refine classes with F1 below this |
| `max_refined_per_class` | 16 | Maximum samples to generate per weak class |
| `include_zero_support` | False | Also generate for classes with no test samples |
| `batch_size` | 8 | Samples per LLM call |
| `max_temperature` | 0.9 | Generation diversity |

### Choosing the F1 Threshold

```python
# Strict: Only fix the worst problems
refiner = LLMRefiner(api_key=api_key, min_f1_threshold=0.5)

# Moderate: Fix classes that underperform
refiner = LLMRefiner(api_key=api_key, min_f1_threshold=0.8)

# Aggressive: Polish everything that's not perfect
refiner = LLMRefiner(api_key=api_key, min_f1_threshold=0.95)
```

---

## Use Cases

### Use Case 1: "Fix the most confused classes"

**Problem**: Two classes keep getting mixed up.

**Solution**: The refiner automatically detects confusion pairs and generates distinguishing samples.

```python
# After seeing this in error analysis:
#   promotional → news-sharing (5 errors)
#   news-sharing → promotional (3 errors)

refiner = LLMRefiner(
    api_key=api_key,
    min_f1_threshold=0.85,
    max_refined_per_class=24,  # More samples for heavily confused classes
)

refined = refiner.refine(result, existing_data)
```

The refiner will:
1. Identify that `promotional` and `news-sharing` confuse each other
2. Analyze what distinguishes them (promotional has calls-to-action, news-sharing is informational)
3. Generate samples that clearly exhibit each class's unique traits

---

### Use Case 2: "Generate data for underrepresented classes"

**Problem**: Some classes have zero test samples, so the model never learned them properly.

**Solution**: Enable `include_zero_support` to generate initial data for these classes.

```python
# Classes with support=0: question, quote, reaction
refiner = LLMRefiner(
    api_key=api_key,
    include_zero_support=True,   # Generate for zero-support classes too
    max_refined_per_class=32,    # More samples since starting from scratch
)

refined = refiner.refine(result, existing_data)
```

---

### Use Case 3: "Iterative refinement"

**Problem**: One round of refinement isn't enough.

**Solution**: Refine, retrain, analyze, repeat.

```python
for iteration in range(3):
    print(f"\n=== Iteration {iteration + 1} ===")
    
    # Train
    classifier.train(data)
    
    # Analyze
    result = classifier.analyze()
    
    # Check if good enough
    weak_classes = sum(
        1 for task in result["tasks"].values()
        for m in task["class_metrics"].values()
        if m["f1"] < 0.85
    )
    
    if weak_classes == 0:
        print("All classes above threshold!")
        break
    
    print(f"Refining {weak_classes} weak classes...")
    
    # Refine
    refined = refiner.refine(result, data)
    data = refined  # Use refined data for next iteration
```

---

### Use Case 4: "Refine specific tasks only"

**Problem**: You have multiple tasks but only one needs refinement.

**Solution**: Filter the analysis result before refining.

```python
result = classifier.analyze()

# Only keep the task you want to refine
filtered_result = {
    "tasks": {
        "storytelling_style": result["tasks"]["storytelling_style"]
    }
}

refined = refiner.refine(filtered_result, existing_data)
```

---

### Use Case 5: "Aggressive refinement for production"

**Problem**: You need near-perfect performance before deployment.

**Solution**: Lower threshold, increase samples, multiple iterations.

```python
refiner = LLMRefiner(
    api_key=api_key,
    min_f1_threshold=0.95,       # Refine anything below 95%
    max_refined_per_class=48,    # Generate lots of samples
    include_zero_support=True,   # Cover all classes
)

# Multiple refinement rounds
for _ in range(3):
    classifier.train(data)
    result = classifier.analyze()
    data = refiner.refine(result, data)

classifier.save("./production_model")
```

---

## What the Refiner Generates

The refiner doesn't just generate random samples. It creates **contrastive** samples based on error analysis.

### Example: promotional vs news-sharing confusion

**Error sample**:
```
Text: "اعلان: دورتنا الجديدة لتعليم البرمجة بدأت"
True: promotional → Predicted: news-sharing (50%)
```

**Refiner analysis**:
```json
{
  "promotional": {
    "confused_with": ["news-sharing"],
    "differentiators": [
      "Contains explicit calls-to-action (buy, subscribe, join)",
      "Focuses on benefits to the reader",
      "Uses persuasive language"
    ],
    "emphasize": [
      "Direct promotional intent",
      "Product or service focus"
    ],
    "avoid": [
      "Neutral informational tone",
      "Third-party news reporting style"
    ]
  }
}
```

**Generated samples**:
```
"سجّل الآن في دورتنا المميزة واحصل على خصم 50%!"
"Join our exclusive program - limited spots available!"
"احجز مقعدك اليوم واستفد من العرض الخاص"
```

These samples clearly exhibit promotional traits that distinguish them from news-sharing.

---

## Best Practices

1. **Analyze before refining** → Don't blindly generate data. Understand what's actually wrong.

2. **Start with a reasonable threshold** → 0.8-0.85 is usually a good starting point. Too low ignores problems, too high wastes tokens.

3. **Check the generated data** → Review a sample of refined outputs to ensure quality.

4. **Iterate thoughtfully** → 2-3 refinement rounds are usually enough. Diminishing returns after that.

5. **Keep original data** → Refined data augments your existing data, doesn't replace it.

6. **Monitor token usage** → The refiner logs call count and tokens. Watch your API costs.

---

## Quick Reference

| I want to... | Code |
|--------------|------|
| Basic refinement | `refiner.refine(result, data)` |
| Stricter threshold | `LLMRefiner(api_key=key, min_f1_threshold=0.7)` |
| More samples per class | `LLMRefiner(api_key=key, max_refined_per_class=32)` |
| Include zero-support classes | `LLMRefiner(api_key=key, include_zero_support=True)` |
| Custom save path | `LLMRefiner(api_key=key, save_path="./refined")` |
| JSONL output | `LLMRefiner(api_key=key, save_format="jsonl")` |

---

## Complete Example

```python
import os
from rapidfit import MultiheadClassifier, LLMRefiner, SaveFormat

# Initial training data
seed_data = {
    "storytelling_style": [
        {"text": "Breaking: New policy announced today", "label": "news-sharing"},
        {"text": "Buy now and save 50%!", "label": "promotional"},
        {"text": "This made my day 😂", "label": "humor-meme"},
        # ... more samples
    ]
}

# Step 1: Initial augmentation and training
from rapidfit import LLMAugmenter

augmenter = LLMAugmenter(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_samples_per_task=128,
)
augmented = augmenter.augment(seed_data)

classifier = MultiheadClassifier()
classifier.train(augmented)

# Step 2: Analyze errors
result = classifier.analyze()
classifier.display(result)

# Step 3: Refine weak classes
refiner = LLMRefiner(
    api_key=os.getenv("OPENAI_API_KEY"),
    min_f1_threshold=0.85,
    max_refined_per_class=16,
    save_path="./saved_refined",
    save_format=SaveFormat.JSONL,
)

# Load existing data for refinement
from rapidfit import DataSaver
saver = DataSaver("./saved", SaveFormat.JSONL)
existing_data = saver.load_all()

refined = refiner.refine(result, existing_data)

# Step 4: Retrain with enriched data (load from saved_refined directory)
classifier.train(data_save_dir="./saved_refined")

# Step 5: Verify improvement
new_result = classifier.analyze()
classifier.display(new_result)

# Compare before/after
for task in result["tasks"]:
    old_acc = result["tasks"][task]["accuracy"]
    new_acc = new_result["tasks"][task]["accuracy"]
    print(f"{task}: {old_acc:.2%} → {new_acc:.2%}")
```

---

## What's Next?

- Check [ERROR_ANALYSIS_GUIDE.md](ERROR_ANALYSIS_GUIDE.md) for understanding your model's failures
- Check [HOW_MULTIHEAD_CLASSIFIER.md](HOW_MULTIHEAD_CLASSIFIER.md) for training configuration
- Check [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for production deployment
