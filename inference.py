#!/usr/bin/env python3
"""
Inference script: Load saved model and predict on new messages
"""

import argparse
from pathlib import Path
from src.pipeline import FraudDetectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Predict fraud/spam on new messages")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved model (.pkl file)")
    parser.add_argument("--text", type=str, help="Single message text")
    parser.add_argument("--file", type=Path, help="File with messages (one per line)")
    
    args = parser.parse_args()

    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        return

    # Load model
    print(f"🔄 Loading model from {args.model}")
    pipeline = FraudDetectionPipeline.load_model(args.model)
    print("✅ Model loaded successfully\n")

    # Prepare messages
    messages = []
    if args.text:
        messages = [args.text]
    elif args.file:
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            return
        with open(args.file, encoding="utf-8") as f:
            messages = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("📝 Enter messages (one per line, empty line to quit):")
        while True:
            msg = input("> ").strip()
            if not msg:
                break
            messages.append(msg)

    if not messages:
        print("⚠️  No messages to predict")
        return

    # Predict
    print(f"\n📊 Predicting on {len(messages)} message(s)...\n")
    results = pipeline.predict(messages)

    # Display results
    for idx, (_, row) in enumerate(results.iterrows(), 1):
        text = row["text"][:60] + "..." if len(row["text"]) > 60 else row["text"]
        fraud_prob = row["fraud_probability"]
        severity = row["severity"]
        risk_score = row["risk_score"]

        icon = "🚨" if severity == "HIGH" else "⚠️ " if severity == "MEDIUM" else "✅"
        print(f"{icon} Message {idx}:")
        print(f"   Text: {text}")
        print(f"   Fraud Probability: {fraud_prob:.2%}")
        print(f"   Risk Score: {risk_score:.2f}")
        print(f"   Severity: {severity}")
        print()


if __name__ == "__main__":
    main()
