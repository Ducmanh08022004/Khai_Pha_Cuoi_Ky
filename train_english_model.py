#!/usr/bin/env python3
"""
Quick start script to train fraud detection model on English datasets with progress tracking.
Usage: python train_english_model.py
"""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import time


def print_header():
    """Print training header."""
    print("\n" + "=" * 80)
    print("🚀 FRAUD DETECTION MODEL TRAINING - ENGLISH DATASETS")
    print("=" * 80)


def print_stage(stage_num: int, stage_name: str, total_stages: int = 6):
    """Print current stage."""
    print(f"\n[Stage {stage_num}/{total_stages}] {stage_name}")


def main():
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset" / "processed"
    output_dir = project_root / "results"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if normalized datasets exist
    sms_file = dataset_dir / "spam_normalized.csv"
    email_file = dataset_dir / "phishing_email_normalized.csv"
    
    if not sms_file.exists():
        print(f"❌ SMS dataset not found: {sms_file}")
        print("   Run: python normalize_datasets.py")
        sys.exit(1)
    
    if not email_file.exists():
        print(f"❌ Email dataset not found: {email_file}")
        print("   Run: python normalize_datasets.py")
        sys.exit(1)
    
    output_file = output_dir / "results_english.json"
    
    print_header()
    
    print(f"\n📁 Datasets:")
    print(f"   SMS:   {sms_file.name} (5,572 messages)")
    print(f"   Email: {email_file.name} (82,474 emails)")
    print(f"\n📤 Output: {output_file}")
    
    # Simulate progress for preparation stages
    print_stage(1, "Preparing datasets", 6)
    with tqdm(total=100, desc="Validating datasets", unit="%", ncols=80) as pbar:
        for _ in range(5):
            time.sleep(0.1)
            pbar.update(20)
    
    print("\n" + "-" * 80)
    print_stage(2, "Running full training pipeline", 6)
    
    # Run main.py with normalized datasets
    cmd = [
        sys.executable,
        "main.py",
        "--sms-data", str(sms_file),
        "--email-data", str(email_file),
        "--output", str(output_file),
        "--n-clusters", "5",
    ]
    
    print("\n📊 Training in progress...\n")
    
    # Create progress bar for training
    with tqdm(total=100, desc="Training model", unit="%", ncols=80, 
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}') as pbar:
        
        # Run subprocess
        result = subprocess.run(
            cmd, 
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        # Parse output for progress indicators
        output_lines = result.stdout.split('\n') if result.stdout else []
        
        # Look for stage indicators
        stage_keywords = {
            "[info] Stage 1/6": 15,
            "[info] Stage 2/6": 25,
            "[info] Stage 3/6": 40,
            "[info] Stage 4/6": 60,
            "[info] Stage 5/6": 80,
            "[info] Finalizing": 95,
        }
        
        last_progress = 0
        for line in output_lines:
            for keyword, progress in stage_keywords.items():
                if keyword in line and progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
        
        # Complete the progress bar
        if last_progress < 100:
            pbar.update(100 - last_progress)
    
    print("\n" + "=" * 80)
    
    if result.returncode == 0:
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📊 Results saved to: {output_file}")
        
        # Try to show results summary
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
            
            if 'roc_auc' in results and results['roc_auc'] is not None:
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
        except Exception as e:
            pass
        
        return 0
    else:
        print("❌ TRAINING FAILED")
        print("=" * 80)
        
        if result.stderr:
            print("\nError output:")
            print(result.stderr[-500:])  # Last 500 chars
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
