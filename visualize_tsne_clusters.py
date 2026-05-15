#!/usr/bin/env python3
"""
Script to visualize fraud clusters using t-SNE dimensionality reduction.

Creates visualization similar to the provided t-SNE plot showing cluster distributions.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.pipeline import FraudDetectionPipeline
from src.preprocessing import preprocess_dataframe
from src.features import transform_feature_bundle
from src.clustering import assign_clusters


def load_model(model_path: Path):
    """Load trained model."""
    print(f"📦 Loading model: {model_path.name}")
    try:
        pipeline = FraudDetectionPipeline.load_model(model_path)
        print("✅ Model loaded successfully\n")
        return pipeline
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def load_datasets():
    """Load SMS and email datasets."""
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset" / "processed"
    
    print("📁 Loading datasets...")
    
    # Load SMS data
    sms_file = dataset_dir / "spam_normalized.csv"
    if not sms_file.exists():
        print(f"❌ SMS dataset not found: {sms_file}")
        sys.exit(1)
    sms_df = pd.read_csv(sms_file, encoding="utf-8", on_bad_lines="skip")
    print(f"   ✓ SMS: {len(sms_df)} rows")
    
    # Load Email data
    email_file = dataset_dir / "phishing_email_downsampled.csv"
    if not email_file.exists():
        email_file = dataset_dir / "phishing_email_normalized.csv"
    
    if not email_file.exists():
        print(f"❌ Email dataset not found")
        sys.exit(1)
    
    email_df = pd.read_csv(email_file, encoding="utf-8", on_bad_lines="skip")
    print(f"   ✓ Email: {len(email_df)} rows from {email_file.name}")
    
    return sms_df, email_df


def prepare_data(sms_df, email_df):
    """Combine and preprocess datasets."""
    print("\n🔄 Preparing data...")
    
    # Handle SMS text column
    if 'text' in sms_df.columns:
        text_col = 'text'
    elif 'SMS' in sms_df.columns:
        text_col = 'SMS'
    else:
        text_col = sms_df.columns[0]
    
    sms_df_prep = sms_df[[text_col]].copy()
    sms_df_prep.columns = ['text']
    
    # Handle Email text column
    if 'Email Text' in email_df.columns:
        text_col = 'Email Text'
    elif 'text' in email_df.columns:
        text_col = 'text'
    else:
        text_col = email_df.columns[0]
    
    email_df_prep = email_df[[text_col]].copy()
    email_df_prep.columns = ['text']
    
    # Combine
    combined_df = pd.concat([sms_df_prep, email_df_prep], ignore_index=True)
    combined_df = combined_df.dropna(subset=['text']).reset_index(drop=True)
    
    print(f"   ✓ Combined: {len(combined_df)} total samples")
    return combined_df


def extract_features(pipeline, data_df):
    """Extract features using trained feature bundle."""
    print("\n📊 Extracting features...")
    
    prepared_df = preprocess_dataframe(data_df, text_col='text', language='en')
    feature_bundle = pipeline.artifacts.feature_bundle
    X_matrix = transform_feature_bundle(feature_bundle, prepared_df, text_col='clean_text')
    
    print(f"   ✓ Feature matrix shape: {X_matrix.shape}")
    return X_matrix


def apply_tsne(X_matrix, perplexity=30, max_iter=1000, random_state=42):
    """Apply t-SNE dimensionality reduction."""
    print("\n🔍 Applying t-SNE dimensionality reduction...")
    print(f"   This may take a minute or two, please wait...\n")
    
    # Convert sparse to dense if needed
    if hasattr(X_matrix, 'toarray'):
        X_dense = X_matrix.toarray()
    else:
        X_dense = np.asarray(X_matrix)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    
    X_tsne = tsne.fit_transform(X_dense)
    print(f"   ✓ t-SNE embedding shape: {X_tsne.shape}\n")
    
    return X_tsne


def plot_tsne_clusters(X_tsne, cluster_labels, pipeline, output_path, title="t-SNE Visualization of Fraud Clusters"):
    """Create t-SNE visualization with clusters."""
    print("📊 Creating t-SNE visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Create colormap
    if n_clusters <= 10:
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_clusters))
    else:
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for idx, cluster_id in enumerate(sorted(unique_clusters)):
        mask = cluster_labels == cluster_id
        scatter = ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=[colors[idx]],
            label=f'{cluster_id}',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.3
        )
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create legend
    legend = ax.legend(
        title='Cluster ID',
        loc='upper left',
        fontsize=10,
        title_fontsize=11,
        framealpha=0.95
    )
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Plot saved to: {output_path}")
    plt.close()


def print_cluster_info(cluster_labels, pipeline):
    """Print cluster information."""
    print("\n📊 Cluster Information:")
    print("-" * 70)
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    total_samples = len(cluster_labels)
    
    for cluster_id, count in zip(unique_clusters, counts):
        percentage = (count / total_samples) * 100
        risk_score = pipeline.artifacts.cluster_risk_map.get(int(cluster_id), 0.0)
        print(f"   Cluster {int(cluster_id):2d}: {count:6d} samples ({percentage:6.2f}%) | Risk Score: {risk_score:.4f}")
    
    print("-" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize fraud clusters using t-SNE")
    parser.add_argument("--model", type=Path, help="Path to saved model (.pkl file)")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (default: 30)")
    parser.add_argument("--max-iter", type=int, default=1000, help="t-SNE max iterations (default: 1000)")
    parser.add_argument("--output", type=Path, help="Output image path (default: results/tsne_clusters.png)")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find latest model if not specified
    if args.model is None:
        models_dir = project_root / "models"
        if not models_dir.exists():
            print("❌ No models directory found. Train model first with: python train_optimized.py")
            return
        
        model_files = sorted(models_dir.glob("*.pkl"))
        if not model_files:
            print("❌ No model files found in models/ directory")
            return
        
        args.model = model_files[-1]
        print(f"📌 Using latest model: {args.model.name}\n")
    
    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Set output path
    if args.output is None:
        args.output = output_dir / "tsne_clusters.png"
    
    print("=" * 80)
    print("🎨 T-SNE CLUSTER VISUALIZATION")
    print("=" * 80 + "\n")
    
    # Load and process data
    pipeline = load_model(args.model)
    sms_df, email_df = load_datasets()
    combined_df = prepare_data(sms_df, email_df)
    X_matrix = extract_features(pipeline, combined_df)
    
    # Get cluster labels
    cluster_labels = assign_clusters(pipeline.artifacts.cluster_model, X_matrix)
    
    # Apply t-SNE
    X_tsne = apply_tsne(X_matrix, perplexity=args.perplexity, max_iter=args.max_iter)
    
    # Print cluster info
    print_cluster_info(cluster_labels, pipeline)
    
    # Create visualization
    plot_tsne_clusters(X_tsne, cluster_labels, pipeline, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
