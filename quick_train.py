#!/usr/bin/env python3
"""
Ultra-fast training script for quick testing.
- 10K rows (30% of downsampled data)
- 50 RF trees
- 1500 TF-IDF features
- Skip rules
Expected time: 2-3 minutes
"""

import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset" / "processed"
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check files
    sms_file = dataset_dir / "spam_normalized.csv"
    email_file = dataset_dir / "phishing_email_downsampled.csv"
    
    if not sms_file.exists() or not email_file.exists():
        print("❌ Dataset files not found")
        return 1
    
    print("\n" + "=" * 80)
    print("⚡ ULTRA-FAST TRAINING (for quick testing)")
    print("=" * 80)
    print("\nConfiguration:")
    print("  • Dataset: 10K samples (30% of downsampled)")
    print("  • RF Trees: 50")
    print("  • TF-IDF Features: 1500")
    print("  • Mining Rules: OFF")
    print("  Expected time: 2-3 minutes\n")
    
    cmd = [
        sys.executable,
        "main.py",
        "--sms-data", str(sms_file),
        "--email-data", str(email_file),
        "--output", str(output_dir / "results_quick.json"),
        "--n-clusters", "3",
        "--n-estimators", "50",
        "--max-features", "1500",
        "--sample-size", "0.3",
        "--skip-rules",
    ]
    
    print("Starting training...\n")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ QUICK TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Results saved to: {output_dir / 'results_quick.json'}\n")
        return 0
    else:
        print("\n❌ Training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
