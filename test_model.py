#!/usr/bin/env python3
"""
Test script: Evaluate saved model on sample messages
"""

import argparse
from pathlib import Path
from src.pipeline import FraudDetectionPipeline


# Sample test messages
TEST_MESSAGES = {
    "spam_sms": [
        "Congratulations! You won $1000. Click here to claim your prize now!",
        "Verify your account immediately. Your password expires today!",
        "URGENT: Confirm your bank details or your account will be closed!",
        "FREE iPhone 13! Limited offer. Click link: http://bit.ly/x123y",
        "You have been selected for a gift card worth $500!",
    ],
    "phishing_email": [
        "Dear Customer, Please confirm your Amazon account by clicking here immediately",
        "ACTION REQUIRED: Unusual activity detected. Verify your PayPal account now",
        "Chase Bank Security Alert - Confirm your login credentials urgently",
        "Important: Your Bank of America account needs verification within 24 hours",
        "RE: Your payment failed - Update your credit card information here",
    ],
    "legitimate_sms": [
        "Hi, this is your appointment reminder for tomorrow at 2pm",
        "Your order has been shipped. Tracking number: ABC123456",
        "Thanks for your purchase. Receipt: #2024-05-14-xyz",
        "Welcome to our store! Check out our new products this week",
        "Your flight confirmation: BA123 departing May 20 at 10:30",
    ],
    "legitimate_email": [
        "Team meeting scheduled for Friday at 3 PM in conference room B",
        "Project update: All deliverables completed on schedule",
        "Welcome aboard! Here is your employee handbook and benefits information",
        "Monthly report: Sales exceeded targets by 15% this quarter",
        "Thank you for attending our webinar. Here are the slides",
    ],
}


def test_model(model_path: Path):
    """Load and test model on sample messages."""
    
    print("=" * 80)
    print("🧪 MODEL TEST SUITE")
    print("=" * 80)
    
    # Load model
    print(f"\n📦 Loading model: {model_path.name}")
    try:
        pipeline = FraudDetectionPipeline.load_model(model_path)
        print("✅ Model loaded successfully\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Test each category
    all_results = []
    for category, messages in TEST_MESSAGES.items():
        print(f"\n{'─' * 80}")
        print(f"📋 Testing: {category.upper().replace('_', ' ')}")
        print(f"{'─' * 80}")
        
        results = pipeline.predict(messages)
        
        for idx, (_, row) in enumerate(results.iterrows(), 1):
            text = row["text"][:55] + "..." if len(row["text"]) > 55 else row["text"]
            fraud_prob = row["fraud_probability"]
            severity = row["severity"]
            
            # Determine if prediction is correct
            is_spam_category = "spam" in category or "phishing" in category
            is_correct = (fraud_prob > 0.5 and is_spam_category) or (fraud_prob <= 0.5 and not is_spam_category)
            status = "✅" if is_correct else "❌"
            
            print(f"{status} {idx}. {text}")
            print(f"   Fraud Prob: {fraud_prob:.1%} | Severity: {severity:^7} | Cluster: {int(row['cluster_id'])}")
            
            all_results.append({
                "category": category,
                "fraud_prob": fraud_prob,
                "is_spam": is_spam_category,
                "is_correct": is_correct
            })
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    
    total = len(all_results)
    correct = sum(1 for r in all_results if r["is_correct"])
    accuracy = correct / total * 100
    
    # By category
    print(f"\n📈 Accuracy by Category:")
    for category in TEST_MESSAGES.keys():
        cat_results = [r for r in all_results if r["category"] == category]
        cat_correct = sum(1 for r in cat_results if r["is_correct"])
        cat_accuracy = cat_correct / len(cat_results) * 100
        print(f"  • {category:25} {cat_accuracy:6.1f}% ({cat_correct}/{len(cat_results)})")
    
    # Overall
    print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Average fraud probability
    spam_probs = [r["fraud_prob"] for r in all_results if r["is_spam"]]
    legit_probs = [r["fraud_prob"] for r in all_results if not r["is_spam"]]
    
    if spam_probs:
        print(f"📊 Avg fraud probability for spam/phishing: {sum(spam_probs)/len(spam_probs):.1%}")
    if legit_probs:
        print(f"📊 Avg fraud probability for legitimate: {sum(legit_probs)/len(legit_probs):.1%}")
    
    print(f"\n{'=' * 80}\n")


def test_interactive(model_path: Path):
    """Interactive mode: test on user-provided messages."""
    
    print("=" * 80)
    print("🧪 INTERACTIVE MODEL TEST")
    print("=" * 80)
    print("\n📦 Loading model...")
    
    try:
        pipeline = FraudDetectionPipeline.load_model(model_path)
        print("✅ Model loaded successfully\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print("📝 Enter messages to test (empty line to exit):\n")
    
    while True:
        msg = input("👉 Message: ").strip()
        if not msg:
            break
        
        result = pipeline.predict([msg]).iloc[0]
        fraud_prob = result["fraud_probability"]
        severity = result["severity"]
        risk_score = result["risk_score"]
        
        print(f"\n📊 Prediction:")
        print(f"  • Fraud Probability: {fraud_prob:.1%}")
        print(f"  • Risk Score: {risk_score:.2f}")
        print(f"  • Severity: {severity}")
        
        if severity == "high":
            print(f"  ⚠️  HighRISK - Likely SPAM/PHISHING")
        elif severity == "medium":
            print(f"  ⚠️  MEDIUM RISK - Suspicious")
        else:
            print(f"  ✅ SAFE - Likely legitimate")
        print()


def main():
    parser = argparse.ArgumentParser(description="Test fraud detection model")
    parser.add_argument("--model", type=Path, help="Path to saved model (.pkl file)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Find latest model if not specified
    if args.model is None:
        models_dir = Path("models")
        if not models_dir.exists():
            print("❌ No models directory found. Train model first with: python train_optimized.py balanced")
            return
        
        model_files = sorted(models_dir.glob("*.pkl"))
        if not model_files:
            print("❌ No model files found in models/ directory")
            return
        
        args.model = model_files[-1]  # Latest model
        print(f"📌 Using latest model: {args.model.name}\n")
    
    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    if args.interactive:
        test_interactive(args.model)
    else:
        test_model(args.model)


if __name__ == "__main__":
    main()
