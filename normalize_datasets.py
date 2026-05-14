"""
Script to normalize SMS and Email datasets for project compatibility.
- Standardizes column names
- Converts numeric labels to text labels
- Removes outliers (very long emails)
- Saves cleaned datasets to output folder
"""

import pandas as pd
from pathlib import Path
import sys


def normalize_sms_dataset(input_path: str, output_path: str) -> None:
    """
    Normalize SMS dataset:
    - Rename: v1 -> label, v2 -> text
    - Keep only useful columns
    - Remove rows with missing text/label
    """
    print("\n" + "=" * 80)
    print("NORMALIZING SMS DATASET")
    print("=" * 80)

    df = pd.read_csv(input_path, encoding="latin1", on_bad_lines="skip")
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Keep only v1 and v2
    df = df[["v1", "v2"]].copy()

    # Rename columns
    df.columns = ["label", "text"]

    # Remove rows with missing values
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    print(f"After removing missing values: {len(df)} rows (removed {before - len(df)})")

    # Ensure label values are standardized
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved to: {output_path}")


def normalize_email_dataset(input_path: str, output_path: str) -> None:
    """
    Normalize Email dataset:
    - Rename: text_combined -> Email Text, label -> Email Type
    - Convert numeric labels to text: 0 -> 'Legitimate', 1 -> 'Phishing'
    - Remove very long outliers (> 100K characters)
    - Remove rows with missing values
    """
    print("\n" + "=" * 80)
    print("NORMALIZING EMAIL DATASET")
    print("=" * 80)

    df = pd.read_csv(input_path, encoding="utf-8", on_bad_lines="skip")
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Rename columns
    df = df[["text_combined", "label"]].copy()
    df.columns = ["Email Text", "Email Type"]

    print(f"\nText length statistics:")
    print(df["Email Text"].str.len().describe())

    # Remove very long outliers (> 100K chars - likely corrupted)
    before = len(df)
    df = df[df["Email Text"].str.len() <= 100000]
    print(f"\nRemoved {before - len(df)} emails longer than 100K chars")

    # Remove rows with missing values
    before = len(df)
    df = df.dropna(subset=["Email Text", "Email Type"])
    df = df[df["Email Text"].str.strip() != ""]
    print(f"After removing missing values: {len(df)} rows (removed {before - len(df)})")

    # Convert numeric labels to text
    label_mapping = {
        0: "Legitimate",
        "0": "Legitimate",
        1: "Phishing",
        "1": "Phishing",
    }
    df["Email Type"] = df["Email Type"].map(label_mapping)

    # Remove any NaN labels that couldn't be mapped
    df = df.dropna(subset=["Email Type"])

    print(f"\nLabel distribution:")
    print(df["Email Type"].value_counts())

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved to: {output_path}")


def main():
    """Normalize both datasets"""
    # Setup paths
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    output_dir = project_root / "dataset" / "processed"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("DATASET NORMALIZATION SCRIPT")
    print("=" * 80)

    # Normalize SMS
    sms_input = dataset_dir / "sms" / "spam.csv"
    sms_output = output_dir / "spam_normalized.csv"

    if not sms_input.exists():
        print(f"❌ SMS input file not found: {sms_input}")
        sys.exit(1)

    normalize_sms_dataset(str(sms_input), str(sms_output))

    # Normalize Email
    email_input = dataset_dir / "email" / "phishing_email.csv"
    email_output = output_dir / "phishing_email_normalized.csv"

    if not email_input.exists():
        print(f"❌ Email input file not found: {email_input}")
        sys.exit(1)

    normalize_email_dataset(str(email_input), str(email_output))

    # Summary
    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {sms_output}")
    print(f"  - {email_output}")
    print(f"\nTo run the pipeline:")
    print(f"  python main.py \\")
    print(f"    --sms-data {sms_output} \\")
    print(f"    --email-data {email_output} \\")
    print(f"    --output results.json")


if __name__ == "__main__":
    main()
