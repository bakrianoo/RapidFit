"""Example: Augment data and train a multihead classifier end-to-end."""

import os

from rapidfit import LLMAugmenter, MultiheadClassifier, SaveFormat


def main():
    # Seed data for multiple classification tasks
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

    # Step 1: Augment data
    print("=" * 50)
    print("Step 1: Data Augmentation")
    print("=" * 50)

    augmenter = LLMAugmenter(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1-mini"),
        max_samples_per_task=64,
        batch_size=8,
        save_path="./saved",
        save_format=SaveFormat.JSONL,
        save_incremental=True,
    )

    augment_result = augmenter.augment(seed_data)

    # Step 2: Train classifier
    print("\n" + "=" * 50)
    print("Step 2: Training Classifier")
    print("=" * 50)

    classifier = MultiheadClassifier({
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 16,
        "epochs": 5,
        "freeze_epochs": 2,
        "patience": 2,
        "output_dir": "./training_output",
    })

    # Train directly on augment result (loads from saved files)
    classifier.train(augment_result)

    # Save model
    classifier.save("./model")

    # Step 3: Test predictions
    print("\n" + "=" * 50)
    print("Step 3: Testing Predictions")
    print("=" * 50)

    test_texts = [
        "I absolutely love this!",
        "This is frustrating.",
        "Feeling nostalgic today.",
    ]

    for task in classifier.tasks:
        print(f"\nTask: {task}")
        predictions = classifier.predict(test_texts, task)
        for text, pred in zip(test_texts, predictions):
            print(f"  '{text}' -> {pred['label']} ({pred['confidence']:.2%})")


if __name__ == "__main__":
    main()
