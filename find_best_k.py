#!/usr/bin/env python3
"""
Script to find the best k value for clustering using the downsampled dataset.

This script:
1. Loads the downsampled email dataset and SMS dataset
2. Preprocesses and extracts features
3. Tests k values from 2 to 20
4. Evaluates using silhouette score and inertia (elbow method)
5. Visualizes results and recommends the best k
"""

import sys
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.preprocessing import normalize_text
from src.features import build_text_vectorizer, build_feature_bundle
from src.clustering import choose_best_k


def load_datasets():
    """Load SMS and email downsampled datasets."""
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset" / "processed"
    
    print("📁 Loading datasets...")
    
    # Load SMS data
    sms_file = dataset_dir / "spam_normalized.csv"
    if not sms_file.exists():
        print(f"❌ SMS dataset not found: {sms_file}")
        sys.exit(1)
    sms_df = pd.read_csv(sms_file, encoding="utf-8", on_bad_lines="skip")
    print(f"   ✓ SMS: {len(sms_df)} rows from {sms_file.name}")
    
    # Load Email downsampled data
    email_file = dataset_dir / "phishing_email_downsampled.csv"
    if not email_file.exists():
        print(f"❌ Email downsampled dataset not found: {email_file}")
        print(f"   Run: python downsample_email.py")
        sys.exit(1)
    email_df = pd.read_csv(email_file, encoding="utf-8", on_bad_lines="skip")
    print(f"   ✓ Email (downsampled): {len(email_df)} rows from {email_file.name}")
    
    return sms_df, email_df


def prepare_data(sms_df, email_df):
    """Normalize and combine datasets."""
    print("\n🔄 Preparing data...")
    
    # Normalize SMS
    if 'text' in sms_df.columns:
        text_col = 'text'
    elif 'SMS' in sms_df.columns:
        text_col = 'SMS'
    else:
        text_col = sms_df.columns[0]
    
    sms_df_norm = sms_df[[text_col]].copy()
    sms_df_norm.columns = ['text']
    sms_df_norm['clean_text'] = sms_df_norm['text'].apply(normalize_text)
    
    # Normalize Email
    if 'Email Text' in email_df.columns:
        text_col = 'Email Text'
    elif 'text' in email_df.columns:
        text_col = 'text'
    else:
        text_col = email_df.columns[0]
    
    email_df_norm = email_df[[text_col]].copy()
    email_df_norm.columns = ['text']
    email_df_norm['clean_text'] = email_df_norm['text'].apply(normalize_text)
    
    # Combine datasets
    combined_df = pd.concat([sms_df_norm, email_df_norm], ignore_index=True)
    combined_df = combined_df.dropna(subset=['clean_text']).reset_index(drop=True)
    
    print(f"   ✓ Combined: {len(combined_df)} total samples")
    return combined_df


def extract_features(df, max_features=3500):
    """Extract TF-IDF features from text."""
    print(f"\n📊 Extracting features (max_features={max_features})...")
    
    vectorizer = build_text_vectorizer(max_features=max_features)
    X_text = vectorizer.fit_transform(df['clean_text'])
    
    print(f"   ✓ TF-IDF matrix shape: {X_text.shape}")
    print(f"   ✓ Sparsity: {1 - X_text.nnz / (X_text.shape[0] * X_text.shape[1]):.2%}")
    
    return X_text


def evaluate_k_values(X, k_min=2, k_max=10):
    """Test different k values and return diagnostics."""
    print(f"\n🔍 Testing k values from {k_min} to {k_max}...")
    
    k_values = list(range(k_min, k_max + 1))
    diagnostics = choose_best_k(X, k_values=k_values)
    
    results = {
        'k_values': diagnostics.k_values,
        'inertias': diagnostics.inertias,
        'silhouettes': diagnostics.silhouettes,
    }
    
    print("\n📈 Results:")
    print("-" * 80)
    print(f"{'k':>3} | {'Inertia':>15} | {'Silhouette Score':>15} | {'Status':>20}")
    print("-" * 80)
    
    for k, inertia, silhouette in zip(diagnostics.k_values, diagnostics.inertias, diagnostics.silhouettes):
        status = ""
        if silhouette == max(diagnostics.silhouettes):
            status = "← Best Silhouette"
        print(f"{k:3d} | {inertia:15.2f} | {silhouette:15.4f} | {status}")
    
    print("-" * 80)
    
    return results


def find_best_k(results):
    """Determine the best k value."""
    silhouettes = results['silhouettes']
    k_values = results['k_values']
    
    best_k_silhouette = k_values[np.argmax(silhouettes)]
    
    # Also find elbow point using acceleration (second derivative)
    inertias = results['inertias']
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmax(diffs2) + 1
        best_k_elbow = k_values[elbow_idx]
    else:
        best_k_elbow = k_values[0]
    
    return best_k_silhouette, best_k_elbow


def plot_results(results, output_dir):
    """Create visualization plots."""
    print("\n📊 Creating plots...")
    
    k_values = results['k_values']
    inertias = results['inertias']
    silhouettes = results['silhouettes']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('K-Means Clustering Optimization - Downsampled Dataset', fontsize=14, fontweight='bold')
    
    # Plot 1: Inertia (Elbow Method)
    ax1 = axes[0]
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia', fontsize=11)
    ax1.set_title('Elbow Method - Inertia', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: Silhouette Score
    ax2 = axes[1]
    best_k = k_values[np.argmax(silhouettes)]
    colors = ['red' if k == best_k else 'blue' for k in k_values]
    ax2.bar(k_values, silhouettes, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Score - Higher is Better', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(k_values)
    
    # Highlight best k
    max_silhouette = max(silhouettes)
    ax2.axhline(y=max_silhouette, color='red', linestyle='--', alpha=0.5, label=f'Best: k={best_k}')
    ax2.legend()
    
    plt.tight_layout()
    
    plot_file = output_dir / "k_optimization_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {plot_file}")
    
    plt.close()


def save_results(results, best_k_silhouette, best_k_elbow, output_dir):
    """Save results to JSON file."""
    output_file = output_dir / "k_optimization_results.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "downsampled (SMS + Email)",
        "k_values": results['k_values'],
        "inertias": results['inertias'],
        "silhouettes": results['silhouettes'],
        "best_k_silhouette": int(best_k_silhouette),
        "best_k_elbow": int(best_k_elbow),
        "recommendation": {
            "best_k": int(best_k_silhouette),
            "reason": "Based on highest silhouette score (measures cluster compactness and separation)"
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✓ Results saved to: {output_file}")
    return output_file


def print_recommendations(best_k_silhouette, best_k_elbow, results):
    """Print recommendations."""
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATIONS")
    print("=" * 80)
    
    silhouettes = results['silhouettes']
    max_silhouette = max(silhouettes)
    
    print(f"\n  ✓ Best k by Silhouette Score: {best_k_silhouette}")
    print(f"    - Silhouette Score: {max_silhouette:.4f}")
    print(f"    - Measure: Cluster compactness and separation (0.0-1.0, higher is better)")
    
    print(f"\n  ✓ Best k by Elbow Method: {best_k_elbow}")
    print(f"    - Measure: Point of diminishing returns in inertia reduction")
    
    print(f"\n  💡 Suggestion: Use k = {best_k_silhouette} for training")
    print(f"    - Run: python train_optimized.py balanced --n-clusters {best_k_silhouette}")
    
    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("🔍 FINDING BEST K VALUE FOR CLUSTERING")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    sms_df, email_df = load_datasets()
    
    # Prepare data
    combined_df = prepare_data(sms_df, email_df)
    
    # Extract features
    X = extract_features(combined_df, max_features=3500)
    
    # Evaluate k values
    results = evaluate_k_values(X, k_min=2, k_max=10)
    
    # Find best k
    best_k_silhouette, best_k_elbow = find_best_k(results)
    
    # Create visualizations
    plot_results(results, output_dir)
    
    # Save results
    save_results(results, best_k_silhouette, best_k_elbow, output_dir)
    
    # Print recommendations
    print_recommendations(best_k_silhouette, best_k_elbow, results)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
