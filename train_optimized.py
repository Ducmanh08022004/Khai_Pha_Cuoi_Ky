#!/usr/bin/env python3
"""
Optimized training script with tunable parameters for faster training.
Usage: python train_optimized.py [--fast|--balanced|--accurate]
"""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import time
import argparse


class TrainingConfig:
    """Training configuration presets."""
    
    FAST = {
        "description": "⚡ Fast training (2-3 min)",
        "n_estimators": 50,
        "max_features": 2000,
        "min_support": 0.25,
        "skip_rules": False,
    }
    
    BALANCED = {
        "description": "⚖️ Balanced (5-8 min)",
        "n_estimators": 150,
        "max_features": 3500,
        "min_support": 0.20,
        "skip_rules": False,
    }
    
    ACCURATE = {
        "description": "🎯 Accurate (10-15 min)",
        "n_estimators": 300,
        "max_features": 5000,
        "min_support": 0.15,
        "skip_rules": False,
    }
    
    PRESETS = {
        "fast": FAST,
        "balanced": BALANCED,
        "accurate": ACCURATE,
    }


def print_config(config_name: str, config: dict):
    """Print training configuration."""
    print(f"\n📋 Training Configuration: {config['description']}")
    print("-" * 80)
    print(f"  • Random Forest Trees: {config['n_estimators']}")
    print(f"  • TF-IDF Max Features: {config['max_features']}")
    print(f"  • Apriori Min Support: {config['min_support']}")
    print(f"  • Mining Rules: {'Yes' if not config['skip_rules'] else 'No'}")
    print("-" * 80)


def print_header():
    """Print training header."""
    print("\n" + "=" * 80)
    print("🚀 OPTIMIZED FRAUD DETECTION MODEL TRAINING")
    print("=" * 80)


def print_stage(stage_num: int, stage_name: str, total_stages: int = 6):
    """Print current stage."""
    print(f"\n[Stage {stage_num}/{total_stages}] {stage_name}")


def print_tips():
    """Print optimization tips."""
    print("\n💡 Speed Optimization Tips:")
    print("-" * 80)
    print("  • Use --fast for quick testing & prototyping")
    print("  • Use --balanced for production training (recommended)")
    print("  • Use --accurate for best model performance")
    print("  • Increase TF-IDF features: Better accuracy, slower training")
    print("  • Decrease RF trees: Faster but less accurate")
    print("  • Sample dataset: Create smaller test set first")
    print("-" * 80)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Optimized fraud detection training with tunable parameters"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Training mode (default: balanced)"
    )
    parser.add_argument(
        "--n-trees",
        type=int,
        help="Override number of Random Forest trees"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        help="Override TF-IDF max features"
    )
    parser.add_argument(
        "--sample-size",
        type=float,
        default=1.0,
        help="Use only fraction of data (0.1 = 10%%, 1.0 = 100%%) for quick testing"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email dataset filename (default: phishing_email_normalized.csv, use 'downsampled' for faster training)"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset" / "processed"
    output_dir = project_root / "results"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine email dataset
    if args.email:
        if args.email.lower() == "downsampled":
            email_filename = "phishing_email_downsampled.csv"
        else:
            email_filename = args.email if args.email.endswith(".csv") else f"{args.email}.csv"
    else:
        # Auto-detect: prefer downsampled if it exists, otherwise use full dataset
        if (dataset_dir / "phishing_email_downsampled.csv").exists():
            email_filename = "phishing_email_downsampled.csv"
        else:
            email_filename = "phishing_email_normalized.csv"
    
    # Check if normalized datasets exist
    sms_file = dataset_dir / "spam_normalized.csv"
    email_file = dataset_dir / email_filename
    
    if not sms_file.exists():
        print(f"❌ SMS dataset not found: {sms_file}")
        sys.exit(1)
    
    if not email_file.exists():
        print(f"❌ Email dataset not found: {email_file}")
        print(f"   Tried: {email_file}")
        print(f"   To use downsampled dataset, run: python downsample_email.py")
        sys.exit(1)
    
    # Get config
    config = TrainingConfig.PRESETS[args.mode].copy()
    
    # Override with CLI arguments
    if args.n_trees:
        config["n_estimators"] = args.n_trees
    if args.max_features:
        config["max_features"] = args.max_features
    
    output_file = output_dir / f"results_{args.mode}.json"
    
    print_header()
    print_config(args.mode, config)
    
    print(f"\n📁 Datasets:")
    print(f"   SMS:   {sms_file.name}")
    print(f"   Email: {email_file.name}")
    
    if "downsampled" in email_file.name:
        print(f"\n⚡ Using downsampled email dataset (30K rows) for faster training")
    
    if args.sample_size < 1.0:
        print(f"\n⚠️  Using {args.sample_size*100:.0f}% of data for quick testing")
    
    print(f"\n📤 Output: {output_file}")
    
    # Simulate preparation stage
    print_stage(1, "Preparing datasets", 6)
    with tqdm(total=100, desc="Validating datasets", unit="%", ncols=80, 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}') as pbar:
        for _ in range(5):
            time.sleep(0.05)
            pbar.update(20)
    
    print("\n" + "-" * 80)
    print_stage(2, "Running optimized training pipeline", 6)
    
    # Build command
    cmd = [
        sys.executable,
        "main.py",
        "--sms-data", str(sms_file),
        "--email-data", str(email_file),
        "--output", str(output_file),
        "--n-clusters", "2",
        "--n-estimators", str(config["n_estimators"]),
        "--max-features", str(config["max_features"]),
        "--min-support", str(config["min_support"]),
    ]
    
    if args.sample_size < 1.0:
        cmd.extend(["--sample-size", str(args.sample_size)])
    
    if config["skip_rules"]:
        cmd.append("--skip-rules")
    
    print("\n📊 Training in progress...\n")
    
    # Run subprocess (stream output directly to console)
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
    )
    
    print("\n" + "=" * 80)
    
    if result.returncode == 0:
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📊 Results saved to: {output_file}")
        
        # Try to show results
        try:
            import json
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            print("\n📈 Results Summary:")
            print("-" * 80)
            
            if 'dataset_summary' in results:
                ds = results['dataset_summary']
                print(f"  Total samples: {ds.get('total_rows', 'N/A')}")
                print(f"    - SMS: {ds.get('sms_rows', 'N/A')}")
                print(f"    - Email: {ds.get('email_rows', 'N/A')}")
            
            if 'roc_auc' in results and results['roc_auc']:
                print(f"  ROC-AUC Score: {results['roc_auc']:.4f}")
            
            if 'evaluation' in results and 'weighted avg' in results['evaluation']:
                metrics = results['evaluation']['weighted avg']
                print(f"  Weighted Avg:")
                print(f"    - Precision: {metrics.get('precision', 'N/A'):.4f}")
                print(f"    - Recall: {metrics.get('recall', 'N/A'):.4f}")
                print(f"    - F1-Score: {metrics.get('f1-score', 'N/A'):.4f}")
            
            if 'rules_found' in results:
                print(f"  Association Rules: {results['rules_found']}")
            
            print("-" * 80)
        except Exception:
            pass
        
        print_tips()
        return 0
    else:
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print("Check error messages above for details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
