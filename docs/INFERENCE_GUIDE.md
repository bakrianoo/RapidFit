# Inference Guide

A practical guide to running predictions with RapidFit—from quick prototyping to production deployment.

---

## The Core Idea

You've trained a multihead classifier. Now what?

RapidFit gives you multiple inference paths depending on your needs:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Inference Options                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Development          Production           High Performance     │
│  ───────────          ──────────           ────────────────     │
│  PyTorch (default)    ONNX Runtime         ONNX + INT8          │
│                                                                 │
│  • Easiest            • 2-3x faster        • 3-4x faster        │
│  • Debug friendly     • Cross-platform     • 4x smaller         │
│  • GPU/CPU            • Optimized graphs   • CPU optimized      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Choose based on your situation:**
- Prototyping or debugging? Use PyTorch directly.
- Deploying to production? Export to ONNX.
- Need maximum speed on CPU? Use ONNX with INT8 quantization.

---

## Level 1: PyTorch Inference (Simplest)

This is what you get out of the box. No extra setup, works immediately after training.

### Basic Prediction

```python
from rapidfit import MultiheadClassifier

# Train or load your model
classifier = MultiheadClassifier()
classifier.train(data)

# Predict single task
results = classifier.predict(["I love this product!"], task="sentiment")
# [{"label": "positive", "confidence": 0.94}]

# Predict all tasks at once
all_results = classifier.predict_all_tasks(["I love this product!"])
# {
#     "sentiment": [{"label": "positive", "confidence": 0.94}],
#     "intent": [{"label": "feedback", "confidence": 0.87}],
#     "emotion": [{"label": "happy", "confidence": 0.91}]
# }
```

### Loading a Saved Model

```python
classifier = MultiheadClassifier()
classifier.load("./my_model")

results = classifier.predict(["Great service!"], task="sentiment")
```

### When to Use This

✅ Development and debugging  
✅ Small-scale inference (< 1000 requests/minute)  
✅ When you need to switch between training and inference  
✅ GPU inference with dynamic batching  

❌ High-throughput production  
❌ Deployment to CPU-only servers  
❌ Edge devices or resource-constrained environments  

---

## Level 2: ONNX Export (Production Ready)

ONNX (Open Neural Network Exchange) gives you a portable, optimized model that runs anywhere.

### Why ONNX?

- **Faster**: ONNX Runtime applies graph optimizations automatically
- **Portable**: Run on any platform (Linux, Windows, macOS, ARM)
- **No PyTorch needed**: Lighter deployment footprint
- **Consistent**: Same results across platforms

### Install Export Dependencies

```bash
pip install rapidfit[export]
```

### Export Your Model

```python
from rapidfit import MultiheadClassifier

classifier = MultiheadClassifier()
classifier.load("./my_model")

# Export all tasks
classifier.export_onnx("./onnx_models")
# Creates: ./onnx_models/sentiment.onnx
#          ./onnx_models/sentiment_config.json
#          ./onnx_models/intent.onnx
#          ./onnx_models/intent_config.json
#          ...

# Export specific tasks only
classifier.export_onnx("./onnx_models", tasks=["sentiment"])
```

### Large Models (>2GB)

Models like `xlm-roberta-large` exceed ONNX's 2GB limit. RapidFit auto-detects this and stores weights externally:

```python
classifier.export_onnx("./onnx_models")                    # auto-detect
classifier.export_onnx("./onnx_models", external_data=True)  # force external
```

Output with external data:
```
onnx_models/
├── sentiment.onnx
└── sentiment_weights.bin   # keep both files together
```

### Run with ONNX Runtime

```python
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer and config
tokenizer = AutoTokenizer.from_pretrained("./onnx_models/tokenizer")
config = json.load(open("./onnx_models/sentiment_config.json"))
id2label = {int(k): v for k, v in config["id2label"].items()}

# Load ONNX model
session = ort.InferenceSession("./onnx_models/sentiment.onnx")

# Run inference
text = "I love this product!"
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
logits = session.run(None, dict(inputs))[0]

# Get prediction
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
predicted_id = np.argmax(probs, axis=-1)[0]
label = id2label[predicted_id]
confidence = probs[0, predicted_id]
```

### Batch Inference

```python
texts = ["Great!", "Terrible service", "It's okay I guess"]
inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True)

outputs = session.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
)

predictions = np.argmax(outputs[0], axis=-1)
```

### When to Use This

✅ Production deployment  
✅ CPU inference at scale  
✅ Cross-platform consistency  
✅ Containerized services (Docker, Kubernetes)  

❌ When you need to retrain frequently  
❌ If you need PyTorch-specific features  

---

## Level 3: ONNX with INT8 Quantization (Maximum Speed)

Quantization converts model weights from 32-bit floats to 8-bit integers. The result: 4x smaller models and faster CPU inference.

### Export with Quantization

```python
# One-step export with quantization
classifier.export_onnx("./onnx_models", quantize=True)
# Creates: ./onnx_models/sentiment_int8.onnx
```

### Or Quantize Existing ONNX Models

```python
from rapidfit import quantize_onnx

quantize_onnx("./onnx_models/sentiment.onnx")
# Creates: ./onnx_models/sentiment_int8.onnx

# Custom output path
quantize_onnx(
    "./onnx_models/sentiment.onnx",
    output_path="./optimized/sentiment_fast.onnx"
)
```

### Run the Quantized Model

Same code as regular ONNX—just point to the quantized file:

```python
session = ort.InferenceSession("./onnx_models/sentiment_int8.onnx")
# ... same inference code
```

### What You Get

| Metric | FP32 (Original) | INT8 (Quantized) |
|--------|-----------------|------------------|
| Model size | ~100 MB | ~25 MB |
| CPU inference | 50ms | 15-20ms |
| Accuracy | 100% | 99-100% |

### When to Use This

✅ CPU-only deployment  
✅ Edge devices (Raspberry Pi, mobile)  
✅ Cost-sensitive cloud deployment  
✅ High-throughput services  

❌ GPU inference (use FP16 instead)  
❌ When every 0.1% accuracy matters  

---

## Choosing the Right Approach

### Decision Tree

```
Do you need to retrain frequently?
├── Yes → Use PyTorch inference
└── No → Continue below

Is this for production?
├── No → Use PyTorch inference
└── Yes → Continue below

Are you deploying to GPU?
├── Yes → Use ONNX (FP32)
└── No (CPU) → Use ONNX + INT8
```

### Quick Reference

| Situation | Recommendation |
|-----------|----------------|
| Local development | PyTorch |
| Jupyter notebooks | PyTorch |
| REST API on GPU | ONNX |
| REST API on CPU | ONNX + INT8 |
| Serverless (Lambda, Cloud Functions) | ONNX + INT8 |
| Edge/IoT devices | ONNX + INT8 |
| Mobile apps | ONNX + INT8 |

---

## Performance Tips

### 1. Batch Your Requests

Single requests waste compute. Batch when possible:

```python
# Slow: One at a time
for text in texts:
    result = classifier.predict([text], task="sentiment")

# Fast: All at once  
results = classifier.predict(texts, task="sentiment")
```

### 2. Reuse Sessions

For ONNX, create the session once:

```python
# Bad: Creates new session per request
def predict(text):
    session = ort.InferenceSession("model.onnx")  # Slow!
    return session.run(...)

# Good: Reuse session
session = ort.InferenceSession("model.onnx")

def predict(text):
    return session.run(...)  # Fast!
```

### 3. Use Appropriate Batch Sizes

- **GPU**: Larger batches (32, 64, 128)
- **CPU**: Moderate batches (8, 16, 32)
- **Memory limited**: Small batches (4, 8)

### 4. Truncate Long Texts

Inference time scales with sequence length:

```python
# If you trained with max_length=128, use the same for inference
inputs = tokenizer(text, max_length=128, truncation=True)
```

---

## Common Patterns

### FastAPI Service with ONNX

```python
from fastapi import FastAPI
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

app = FastAPI()

# Load once at startup
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
session = ort.InferenceSession("./onnx/sentiment_int8.onnx")
id2label = {0: "negative", 1: "neutral", 2: "positive"}

@app.post("/predict")
def predict(texts: list[str]):
    inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True)
    logits = session.run(None, dict(inputs))[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    
    return [
        {"label": id2label[idx], "confidence": float(conf)}
        for idx, conf in zip(np.argmax(probs, axis=-1), probs.max(axis=-1))
    ]
```

### Multi-Task Inference

When you need all tasks, load all sessions:

```python
sessions = {
    "sentiment": ort.InferenceSession("./onnx/sentiment_int8.onnx"),
    "intent": ort.InferenceSession("./onnx/intent_int8.onnx"),
    "emotion": ort.InferenceSession("./onnx/emotion_int8.onnx"),
}

def predict_all(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    return {
        task: session.run(None, dict(inputs))[0]
        for task, session in sessions.items()
    }
```

---

## Troubleshooting

### "Import onnx could not be resolved"

Install export dependencies:
```bash
pip install rapidfit[export]
```

### "Model must be trained before export"

You need a trained model:
```python
classifier = MultiheadClassifier()
classifier.load("./my_model")  # Load existing model
classifier.export_onnx("./onnx")
```

### ONNX output differs from PyTorch

This shouldn't happen, but if it does:
1. Ensure you're using the same tokenizer
2. Check that `max_length` matches training config
3. Verify the model was saved after training completed

### Quantized model accuracy dropped significantly

Try exporting without quantization first. If FP32 ONNX works correctly, the issue is quantization sensitivity. Some models (especially small ones) are more sensitive to quantization.

---

## What's Next?

- Check `examples/` for complete working scripts
- Read `HOW_MULTIHEAD_CLASSIFIER.md` for training configuration
- Explore ONNX Runtime's advanced features (execution providers, graph optimizations)
