"""Example: Train a multihead classifier on augmented data."""

import json
from pathlib import Path

from rapidfit import MultiheadClassifier, SeedData


def load_jsonl(path: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # Load augmented data from saved files
    saved_dir = Path("saved")
    
    seed_data: SeedData = {}
    for file in saved_dir.glob("*.jsonl"):
        task_name = file.stem
        seed_data[task_name] = load_jsonl(file)
        print(f"Loaded {len(seed_data[task_name])} samples for '{task_name}'")

    if not seed_data:
        print("No data found in 'saved/' directory.")
        print("Run 'python examples/test_augmentation.py' first to generate data.")
        return

    # Create and train classifier
    classifier = MultiheadClassifier({
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 16,
        "epochs": 5,
        "freeze_epochs": 2,
        "patience": 2,
        "output_dir": "./training_output",
        "save_path": "./model",
    })

    classifier.train(seed_data)

    # Save the trained model
    classifier.save("./model")

    # Test predictions
    test_texts = [
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "It's okay, nothing special.",
    ]

    print("\n--- Predictions ---")
    for task in classifier.tasks:
        print(f"\nTask: {task}")
        predictions = classifier.predict(test_texts, task)
        for text, pred in zip(test_texts, predictions):
            print(f"  '{text[:40]}...' -> {pred['label']} ({pred['confidence']:.2%})")


if __name__ == "__main__":
    main()
