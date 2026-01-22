# How the Multihead Classifier Works

A practical guide to understanding and using RapidFit's multihead classification system.

---

## The Core Idea

Imagine you need to build three classifiers:
- One for **sentiment analysis** (positive/negative/neutral)
- One for **intent detection** (buy/sell/ask/complain)
- One for **emotion analysis** (happy/sad/angry/surprised)

The traditional approach? Train three separate models. That's expensive, slow, and wasteful.

**The multihead approach**: Train one shared "brain" (encoder) that understands language, then attach three lightweight "decision makers" (heads) on top. Each head specializes in its own task while sharing the same understanding of text.

```
        "I love this product!"
                 ↓
    ┌────────────────────────┐
    │    Shared Encoder      │  ← Understands what the text means
    │  (language knowledge)  │
    └───────────┬────────────┘
                ↓
         Text Embedding
         ↙    ↓    ↘
    ┌────┐ ┌────┐ ┌────┐
    │Sent│ │Int │ │Emo │    ← Each head makes its own decision
    └────┘ └────┘ └────┘
       ↓     ↓      ↓
   positive  buy   happy
```

**Why this works better:**
- Faster training (one encoder, not three)
- Smaller model size (shared weights)
- Better results (tasks help each other learn)

---

## Getting Started

### Minimal Setup

```python
from rapidfit import MultiheadClassifier

# Your training data
data = {
    "sentiment": [
        {"text": "I love this!", "label": "positive"},
        {"text": "This is terrible", "label": "negative"},
    ],
    "intent": [
        {"text": "I want to buy", "label": "purchase"},
        {"text": "How does this work?", "label": "question"},
    ],
}

# Train with defaults
classifier = MultiheadClassifier()
classifier.train(data)

# Predict
result = classifier.predict(["Great product!"], task="sentiment")
```

That's it. The defaults are tuned to work well for most cases.

---

## Understanding the Configuration

The configuration is organized into logical groups. You only need to touch what matters for your use case.

```python
from rapidfit import MultiheadConfig, TrainingConfig, HeadConfig

config = MultiheadConfig(
    training=TrainingConfig(...),  # How to train
    head=HeadConfig(...),          # Head architecture
    encoder=EncoderConfig(...),    # Encoder fine-tuning
    loss=LossConfig(...),          # Loss function
    eval=EvalConfig(...),          # When to stop
    pooling="mean",                # How to summarize text
    task_sampling="proportional",  # How to balance tasks
)

classifier = MultiheadClassifier(config)
```

---

## Use Cases & Configurations

### Use Case 1: "I have limited GPU memory"

**Problem**: Your GPU runs out of memory during training.

**Solution**: Use gradient accumulation to simulate larger batches without the memory cost.

```python
from rapidfit import MultiheadConfig, TrainingConfig

config = MultiheadConfig(
    training=TrainingConfig(
        batch_size=4,                      # Small batches fit in memory
        gradient_accumulation_steps=4,     # Accumulate 4 steps = effective batch of 16
    ),
)
```

**What's happening**: Instead of processing 16 samples at once, you process 4 samples four times, then update the model. Same learning, less memory.

---

### Use Case 2: "My dataset is heavily imbalanced"

**Problem**: You have 1000 "positive" samples but only 50 "negative" samples. The model ignores the minority class.

**Solution**: Enable focal loss, which forces the model to pay attention to hard-to-classify examples.

```python
from rapidfit import MultiheadConfig, LossConfig

config = MultiheadConfig(
    loss=LossConfig(
        use_class_weights=True,    # Weight classes by inverse frequency
        use_focal_loss=True,       # Focus on hard examples
        focal_gamma=2.0,           # Higher = more focus on hard examples
    ),
)
```

**What's happening**: 
- Class weights make rare classes "worth more" during training
- Focal loss reduces the contribution of easy examples, so the model focuses on what it gets wrong

---

### Use Case 3: "I have one important task and several helper tasks"

**Problem**: You care most about sentiment, but you also have intent and emotion data. You want the model to prioritize sentiment.

**Solution**: Assign higher weight to your main task.

```python
from rapidfit import MultiheadConfig, LossConfig

config = MultiheadConfig(
    loss=LossConfig(
        task_weights={
            "sentiment": 2.0,    # This task matters twice as much
            "intent": 1.0,
            "emotion": 0.5,      # This is just a helper
        },
    ),
)
```

**What's happening**: The model's loss function weighs sentiment errors more heavily, so it optimizes harder for that task.

---

### Use Case 4: "My tasks have very different amounts of data"

**Problem**: Sentiment has 10,000 samples, intent has 500. The model overfits on intent and underfits on sentiment.

**Solution**: Change how tasks are sampled during training.

```python
from rapidfit import MultiheadConfig

# Option A: Equal sampling (each task gets same number of samples per epoch)
config = MultiheadConfig(task_sampling="equal")

# Option B: Square root sampling (balanced middle ground)
config = MultiheadConfig(task_sampling="sqrt")

# Option C: Proportional (default - use actual data distribution)
config = MultiheadConfig(task_sampling="proportional")
```

**When to use what**:
- `proportional`: Your task sizes are similar
- `sqrt`: Moderate imbalance (2-5x difference)
- `equal`: Extreme imbalance (10x+ difference)

---

### Use Case 5: "I want to prevent overfitting"

**Problem**: Training accuracy is 99%, but validation accuracy is 70%. Classic overfitting.

**Solution**: Increase regularization and use early stopping.

```python
from rapidfit import MultiheadConfig, HeadConfig, EvalConfig, TrainingConfig

config = MultiheadConfig(
    head=HeadConfig(
        dropout=0.3,              # Was 0.2, increase to drop more neurons
    ),
    training=TrainingConfig(
        weight_decay=0.05,        # Was 0.01, stronger L2 regularization
    ),
    eval=EvalConfig(
        patience=2,               # Stop after 2 epochs without improvement
        min_delta=0.005,          # Improvement must be at least 0.5%
        metric="accuracy",        # Use "f1" for F1-based early stopping
    ),
)
```

**What's happening**:
- Dropout randomly disables neurons, forcing the network to not rely on specific pathways
- Weight decay penalizes large weights, keeping the model simpler
- Early stopping prevents training too long

---

### Use Case 6: "I have complex classification tasks"

**Problem**: Your tasks are nuanced. The default single hidden layer in each head isn't enough.

**Solution**: Make the heads deeper and wider.

```python
from rapidfit import MultiheadConfig, HeadConfig

config = MultiheadConfig(
    head=HeadConfig(
        hidden_layers=2,          # Was 1, now 2 layers
        hidden_multiplier=1.5,    # Was 1.0, now 50% wider
        activation="silu",        # SiLU often works better for complex patterns
    ),
)
```

**Trade-off**: Deeper heads learn more complex patterns but need more data. Start simple, add complexity only if needed.

---

### Use Case 7: "I want maximum accuracy and have time"

**Problem**: You want the best possible model and training time isn't a constraint.

**Solution**: Fine-tune more of the encoder and train longer.

```python
from rapidfit import MultiheadConfig, EncoderConfig, TrainingConfig

config = MultiheadConfig(
    encoder=EncoderConfig(
        freeze_epochs=5,          # Train heads only for 5 epochs first
        unfreeze_layers=6,        # Then unfreeze top 6 encoder layers (was 4)
        lr_multiplier=0.05,       # Encoder learns slower than heads
    ),
    training=TrainingConfig(
        epochs=20,                # Train longer
        learning_rate=1e-5,       # Lower learning rate for stability
    ),
)
```

**What's happening**:
- First phase: Only heads train, encoder stays frozen
- Second phase: Top encoder layers unfreeze and fine-tune slowly
- Lower learning rate prevents destroying pre-trained knowledge

---

### Use Case 8: "I need different pooling for my text type"

**Problem**: Your texts are short (tweets, titles) or you care about the beginning.

**Solution**: Change the pooling strategy.

```python
from rapidfit import MultiheadConfig

# For short texts or when first tokens matter most
config = MultiheadConfig(pooling="cls")

# For texts where any part could be important
config = MultiheadConfig(pooling="mean")  # default

# When key information is scattered (picks strongest signals)
config = MultiheadConfig(pooling="max")
```

**Quick guide**:
- `mean`: Best default for most text types
- `cls`: Good for short texts, titles, first-sentence-heavy content
- `max`: Good when you're looking for "presence" of features

---

### Use Case 9: "I want to control train/validation/test splits"

**Problem**: The default 10% test and 10% validation splits don't fit your needs. You want more data for training or a larger test set.

**Solution**: Adjust split sizes in `TrainingConfig`.

```python
from rapidfit import MultiheadConfig, TrainingConfig

config = MultiheadConfig(
    training=TrainingConfig(
        test_size=0.15,   # 15% for test set (default: 0.1)
        val_size=0.15,    # 15% for validation set (default: 0.1)
    ),
)
```

**What's happening**: With defaults (0.1 each), the split is approximately 81% train, 9% validation, 10% test. The validation set is carved from the remaining data after the test split.

**Quick guide**:
- Small dataset (<1000 samples): Use smaller splits (`test_size=0.1`, `val_size=0.1`)
- Large dataset (>10000 samples): Can afford larger test sets (`test_size=0.2`)
- Need more training data: Reduce both (`test_size=0.05`, `val_size=0.05`)

---

### Use Case 10: "I want to optimize for F1 score instead of accuracy"

**Problem**: Your classes are imbalanced, and accuracy is misleading. You want the model to optimize for balanced precision and recall.

**Solution**: Use F1 as the validation metric for early stopping and best model selection.

```python
from rapidfit import MultiheadConfig, EvalConfig

config = MultiheadConfig(
    eval=EvalConfig(
        metric="f1",              # Use macro F1 instead of accuracy
        patience=3,               # Stop after 3 epochs without F1 improvement
    ),
)
```

**What's happening**: During validation, the model computes both accuracy and F1 (macro-averaged across classes). When `metric="f1"`, the trainer saves checkpoints based on F1 score and stops early when F1 plateaus.

**When to use**:
- Imbalanced datasets where accuracy is misleading
- Multi-class problems where you care about all classes equally
- When false positives and false negatives matter equally

**Metric options**:
- `accuracy` (default): Simple accuracy metric
- `f1`: Macro-averaged F1 score across all classes
- `loss`: Use validation loss for model selection

---

## Configuration Quick Reference

| I want to... | Configuration |
|--------------|---------------|
| Use less GPU memory | `training.gradient_accumulation_steps=4` |
| Handle class imbalance | `loss.use_focal_loss=True` |
| Prioritize one task | `loss.task_weights={"main": 2.0}` |
| Balance unequal task sizes | `task_sampling="equal"` |
| Prevent overfitting | `head.dropout=0.3`, `eval.patience=2` |
| Learn complex patterns | `head.hidden_layers=2` |
| Get maximum accuracy | `encoder.unfreeze_layers=6`, `training.epochs=20` |
| Handle short texts | `pooling="cls"` |
| Control data splits | `training.test_size=0.15`, `training.val_size=0.15` |
| Optimize for F1 score | `eval.metric="f1"` |

---

## Recommended Starting Points

### For beginners
```python
# Just use defaults
classifier = MultiheadClassifier()
```

### Changing the base model
```python
config = MultiheadConfig(
    training=TrainingConfig(
        model_name="FacebookAI/xlm-roberta-base",  # Or any HuggingFace model
    ),
)
```

Common model choices:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default, fast, multilingual)
- `FacebookAI/xlm-roberta-base` (stronger multilingual understanding)
- `sentence-transformers/all-MiniLM-L6-v2` (fastest, English only)
- `microsoft/deberta-v3-base` (high accuracy, English)

### For production
```python
config = MultiheadConfig(
    training=TrainingConfig(
        epochs=10,
        batch_size=16,
    ),
    loss=LossConfig(
        use_class_weights=True,
    ),
    eval=EvalConfig(
        patience=3,
    ),
)
```

### For experimentation
```python
config = MultiheadConfig(
    head=HeadConfig(hidden_layers=2, dropout=0.3),
    encoder=EncoderConfig(unfreeze_layers=6, lr_multiplier=0.05),
    loss=LossConfig(use_focal_loss=True, use_class_weights=True),
    pooling="mean",
    task_sampling="sqrt",
)
```

---

## Common Mistakes to Avoid

1. **Starting with complex configurations** → Start simple, add complexity only when you see specific problems.

2. **Unfreezing the entire encoder immediately** → This destroys pre-trained knowledge. Use `freeze_epochs` first.

3. **Setting learning rate too high** → If loss jumps around, lower your learning rate.

4. **Ignoring class imbalance** → Always check your label distribution. Use `use_class_weights=True` by default.

5. **Training too long** → Watch your validation metrics. Use early stopping.

---

## What's Next?

- Check the full API reference for all configuration options
- Look at the examples folder for complete working scripts

