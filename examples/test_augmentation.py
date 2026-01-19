"""Test script for LLM-based data augmentation."""

import os
from rapidfit import LLMAugmenter, WriteMode
from rapidfit.types import SaveFormat


def main():
    seed_data = {
        "sentiment-analysis": [
            {"text": "I absolutely love this product, best purchase ever!", "label": "positive"},
            {"text": "هذا المنتج رائع جداً وأنصح به بشدة", "label": "positive"},
            {"text": "यह सेवा बहुत अच्छी है, मैं बहुत खुश हूं", "label": "positive"},
            {"text": "This is the worst experience I've ever had.", "label": "negative"},
            {"text": "المنتج سيء جداً ولا أنصح بشرائه", "label": "negative"},
            {"text": "यह उत्पाद बहुत खराब है, पैसे की बर्बादी", "label": "negative"},
            {"text": "It's fine, does what it's supposed to do.", "label": "neutral"},
            {"text": "المنتج عادي، لا شيء مميز", "label": "neutral"},
            {"text": "ठीक है, कुछ खास नहीं", "label": "neutral"},
        ],
        "emotion-analysis": [
            {"text": "This makes me so incredibly happy!", "label": "joy"},
            {"text": "أنا سعيد جداً بهذا الخبر الرائع", "label": "joy"},
            {"text": "मुझे बहुत खुशी हुई यह सुनकर", "label": "joy"},
            {"text": "I can't believe they would do this to me.", "label": "anger"},
            {"text": "هذا أمر غير مقبول على الإطلاق", "label": "anger"},
            {"text": "यह बिल्कुल गलत है, मुझे बहुत गुस्सा आ रहा है", "label": "anger"},
            {"text": "I really miss those beautiful days.", "label": "sadness"},
            {"text": "أفتقد تلك الأيام الجميلة كثيراً", "label": "sadness"},
            {"text": "मुझे उन पुराने दिनों की बहुत याद आती है", "label": "sadness"},
        ],
        "intent-detection": [
            {"text": "What's the weather like today?", "label": "weather_query"},
            {"text": "كيف حالة الطقس اليوم؟", "label": "weather_query"},
            {"text": "आज मौसम कैसा है?", "label": "weather_query"},
            {"text": "Book a table for two at 7pm", "label": "reservation"},
            {"text": "أريد حجز طاولة لشخصين الساعة السابعة", "label": "reservation"},
            {"text": "शाम 7 बजे के लिए दो लोगों की टेबल बुक करें", "label": "reservation"},
            {"text": "I need help with my order", "label": "support"},
            {"text": "أحتاج مساعدة بخصوص طلبي", "label": "support"},
            {"text": "मुझे अपने ऑर्डर में मदद चाहिए", "label": "support"},
            {"text": "What are your working hours?", "label": "faq"},
            {"text": "ما هي ساعات العمل لديكم؟", "label": "faq"},
            {"text": "आपके काम के घंटे क्या हैं?", "label": "faq"},
        ],
    }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable")
        return

    augmenter = LLMAugmenter(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        model_id=os.environ.get("OPENAI_MODEL_ID", "gpt-4.1-mini"),
        max_samples_per_task=24,
        batch_size=8,
        max_temperature=0.9,
        save_path="./saved",
        save_format=SaveFormat.JSONL,
        save_incremental=True,
        write_mode=WriteMode.APPEND,
    )

    print("Starting augmentation...")
    results = augmenter.augment(seed_data)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for task, result in results.items():
        print(f"\n{task}:")
        print(f"  path: {result['path']}")
        print(f"  total: {result['stats']['total']}")
        for label, count in sorted(result["stats"]["labels"].items()):
            print(f"    {label}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
