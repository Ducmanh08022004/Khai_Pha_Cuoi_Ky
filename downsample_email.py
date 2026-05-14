#!/usr/bin/env python3
"""
Downsample email dataset to make training faster.
Reduces email dataset from 82K to 30K rows while maintaining class balance.
"""

import pandas as pd
from pathlib import Path


def downsample_email(input_path: str, output_path: str, target_size: int = 30000) -> None:
    """
    Downsample email dataset to target size while maintaining class balance.
    
    Args:
        input_path: Path to normalized email CSV
        output_path: Path to save downsampled CSV
        target_size: Target number of rows (default: 30000)
    """
    print("\n" + "=" * 80)
    print("DOWNSAMPLING EMAIL DATASET")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Original size: {len(df):,} rows")
    print(f"Label distribution (before):")
    print(df["Email Type"].value_counts())
    
    # Calculate samples per class to maintain balance
    class_counts = df["Email Type"].value_counts()
    total_rows = len(df)
    
    # Calculate proportion of each class
    proportions = class_counts / total_rows
    print(f"\nClass proportions: {dict(proportions)}")
    
    # Sample each class proportionally
    samples_per_class = (proportions * target_size).astype(int)
    print(f"Target samples per class: {dict(samples_per_class)}")
    
    # Perform stratified sampling
    downsampled = []
    for class_label, count in samples_per_class.items():
        class_data = df[df["Email Type"] == class_label]
        sampled = class_data.sample(n=min(count, len(class_data)), random_state=42)
        downsampled.append(sampled)
    
    df_downsampled = pd.concat(downsampled, ignore_index=True)
    
    # Shuffle
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nDownsampled size: {len(df_downsampled):,} rows")
    print(f"Label distribution (after):")
    print(df_downsampled["Email Type"].value_counts())
    
    # Calculate reduction
    reduction = (1 - len(df_downsampled) / len(df)) * 100
    print(f"\n✓ Reduced by {reduction:.1f}%")
    
    # Save
    df_downsampled.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✓ Saved to: {output_path}")


def main():
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset" / "processed"
    
    input_file = dataset_dir / "phishing_email_normalized.csv"
    output_file = dataset_dir / "phishing_email_downsampled.csv"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    # Downsample to 30K rows
    downsample_email(str(input_file), str(output_file), target_size=30000)
    
    print("\n" + "=" * 80)
    print("✅ DOWNSAMPLING COMPLETE")
    print("=" * 80)
    print(f"\nTo use downsampled dataset in training:")
    print(f"  python main.py \\")
    print(f"    --sms-data dataset/processed/spam_normalized.csv \\")
    print(f"    --email-data dataset/processed/phishing_email_downsampled.csv")
    print("\nOr with optimized training:")
    print(f"  python train_optimized.py balanced --email phishing_email_downsampled.csv")
    
    return 0


if __name__ == "__main__":
    exit(main())
