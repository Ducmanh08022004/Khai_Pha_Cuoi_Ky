#!/usr/bin/env python3
"""
Script to extract and evaluate clustering metrics from trained model.

Metrics extracted:
1. Silhouette Score - Measures cluster compactness and separation ([-1, 1], higher is better)
2. Davies-Bouldin Index - Ratio of within-cluster to between-cluster distances (lower is better)
3. Calinski-Harabasz Index - Ratio of between-cluster to within-cluster sum of squares (higher is better)
4. Inertia (WCSS) - Within-cluster sum of squares (lower is better)
"""

import sys
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.pipeline import FraudDetectionPipeline
from src.preprocessing import preprocess_dataframe
from src.features import transform_feature_bundle
from src.clustering import assign_clusters


def load_model(model_path: Path):
    """Load saved model."""
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
    
    # Load Email data - try downsampled first, then full
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
    print("\n📊 Extracting features using trained feature bundle...")
    
    # Preprocess text
    prepared_df = preprocess_dataframe(data_df, text_col='text', language='en')
    
    # Transform using trained feature bundle
    feature_bundle = pipeline.artifacts.feature_bundle
    X_matrix = transform_feature_bundle(feature_bundle, prepared_df, text_col='clean_text')
    
    print(f"   ✓ Feature matrix shape: {X_matrix.shape}")
    print(f"   ✓ Sparsity: {1 - X_matrix.nnz / (X_matrix.shape[0] * X_matrix.shape[1]):.2%}")
    
    return X_matrix


def get_clustering_metrics(pipeline, X_matrix):
    """Calculate all clustering metrics."""
    print("\n🔍 Calculating clustering metrics...")
    
    if pipeline.artifacts.cluster_model is None:
        print("❌ No cluster model found in pipeline")
        sys.exit(1)
    
    # Get cluster assignments
    cluster_model = pipeline.artifacts.cluster_model
    cluster_labels = assign_clusters(cluster_model, X_matrix)
    
    # Convert sparse matrix to dense if needed
    if hasattr(X_matrix, 'toarray'):
        X_dense = X_matrix.toarray()
    else:
        X_dense = np.asarray(X_matrix)
    
    print(f"   Number of clusters: {cluster_model.n_clusters}")
    print(f"   Total samples: {len(cluster_labels)}")
    print(f"   Unique clusters: {len(np.unique(cluster_labels))}")
    
    # 1. Silhouette Score
    print("\n   📈 Calculating Silhouette Score...")
    try:
        silhouette = silhouette_score(X_dense, cluster_labels)
        print(f"      ✓ Silhouette Score: {silhouette:.4f}")
    except Exception as e:
        silhouette = None
        print(f"      ⚠️  Error calculating Silhouette Score: {e}")
    
    # 2. Davies-Bouldin Index
    print("   📊 Calculating Davies-Bouldin Index...")
    try:
        davies_bouldin = davies_bouldin_score(X_dense, cluster_labels)
        print(f"      ✓ Davies-Bouldin Index: {davies_bouldin:.4f}")
    except Exception as e:
        davies_bouldin = None
        print(f"      ⚠️  Error calculating Davies-Bouldin Index: {e}")
    
    # 3. Calinski-Harabasz Index
    print("   📈 Calculating Calinski-Harabasz Index...")
    try:
        calinski_harabasz = calinski_harabasz_score(X_dense, cluster_labels)
        print(f"      ✓ Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    except Exception as e:
        calinski_harabasz = None
        print(f"      ⚠️  Error calculating Calinski-Harabasz Index: {e}")
    
    # 4. Inertia (WCSS)
    print("   📉 Extracting Inertia (WCSS)...")
    try:
        inertia = cluster_model.inertia_
        print(f"      ✓ Inertia (WCSS): {inertia:.4f}")
    except Exception as e:
        inertia = None
        print(f"      ⚠️  Error getting Inertia: {e}")
    
    # Silhouette from diagnostics if available
    silhouette_diag = None
    if pipeline.artifacts.cluster_diagnostics:
        diag = pipeline.artifacts.cluster_diagnostics
        if len(diag.silhouettes) > 0:
            # Find silhouette for current n_clusters
            k_idx = diag.k_values.index(cluster_model.n_clusters) if cluster_model.n_clusters in diag.k_values else -1
            if k_idx >= 0:
                silhouette_diag = diag.silhouettes[k_idx]
    
    return {
        'silhouette_score': float(silhouette) if silhouette is not None else None,
        'silhouette_from_diagnostics': float(silhouette_diag) if silhouette_diag is not None else None,
        'davies_bouldin_index': float(davies_bouldin) if davies_bouldin is not None else None,
        'calinski_harabasz_index': float(calinski_harabasz) if calinski_harabasz is not None else None,
        'inertia_wcss': float(inertia) if inertia is not None else None,
        'num_clusters': int(cluster_model.n_clusters),
        'num_samples': int(len(cluster_labels)),
    }


def get_cluster_distribution(pipeline, cluster_labels):
    """Get cluster size distribution."""
    print("\n📊 Cluster Distribution:")
    print("-" * 60)
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    total_samples = len(cluster_labels)
    
    for cluster_id, count in zip(unique_clusters, counts):
        percentage = (count / total_samples) * 100
        risk_score = pipeline.artifacts.cluster_risk_map.get(int(cluster_id), 0.0)
        print(f"   Cluster {int(cluster_id):2d}: {count:6d} samples ({percentage:6.2f}%) | Risk Score: {risk_score:.2f}")
    
    print("-" * 60)


def print_metrics_summary(metrics):
    """Print metrics in a readable format."""
    print("\n" + "=" * 80)
    print("📊 CLUSTERING METRICS SUMMARY")
    print("=" * 80)
    
    print(f"\n🎯 Model Configuration:")
    print(f"   • Number of Clusters (k): {metrics['num_clusters']}")
    print(f"   • Total Samples: {metrics['num_samples']:,}")
    
    print(f"\n📈 Quality Metrics:")
    
    if metrics['silhouette_score'] is not None:
        sil_score = metrics['silhouette_score']
        if sil_score >= 0.5:
            quality = "Excellent"
        elif sil_score >= 0.4:
            quality = "Good"
        elif sil_score >= 0.25:
            quality = "Fair"
        else:
            quality = "Poor"
        print(f"   • Silhouette Score: {sil_score:.4f} ({quality})")
        print(f"     Range: [-1, 1] | Higher is better")
        print(f"     Interpretation: Measures how similar objects are within cluster vs other clusters")
    else:
        print(f"   • Silhouette Score: N/A")
    
    if metrics['davies_bouldin_index'] is not None:
        db_index = metrics['davies_bouldin_index']
        if db_index <= 1.0:
            quality = "Excellent"
        elif db_index <= 1.5:
            quality = "Good"
        elif db_index <= 2.0:
            quality = "Fair"
        else:
            quality = "Poor"
        print(f"   • Davies-Bouldin Index: {db_index:.4f} ({quality})")
        print(f"     Range: [0, ∞] | Lower is better")
        print(f"     Interpretation: Ratio of avg within-cluster distance to between-cluster distance")
    else:
        print(f"   • Davies-Bouldin Index: N/A")
    
    if metrics['calinski_harabasz_index'] is not None:
        ch_index = metrics['calinski_harabasz_index']
        if ch_index >= 1000:
            quality = "Excellent"
        elif ch_index >= 500:
            quality = "Good"
        elif ch_index >= 100:
            quality = "Fair"
        else:
            quality = "Poor"
        print(f"   • Calinski-Harabasz Index: {ch_index:.4f} ({quality})")
        print(f"     Range: [0, ∞] | Higher is better")
        print(f"     Interpretation: Ratio of between-cluster to within-cluster dispersion")
    else:
        print(f"   • Calinski-Harabasz Index: N/A")
    
    if metrics['inertia_wcss'] is not None:
        print(f"   • Inertia (WCSS): {metrics['inertia_wcss']:.4f}")
        print(f"     Range: [0, ∞] | Lower is better")
        print(f"     Interpretation: Sum of squared distances from each point to its cluster center")
    else:
        print(f"   • Inertia (WCSS): N/A")
    
    print("\n" + "=" * 80)


def save_metrics(metrics, cluster_labels, output_dir, model_name):
    """Save metrics to JSON file."""
    output_file = output_dir / "clustering_metrics.json"
    
    # Get cluster distribution
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_dist = {
        int(cid): int(count) for cid, count in zip(unique_clusters, counts)
    }
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "metrics": metrics,
        "cluster_distribution": cluster_dist,
        "metric_descriptions": {
            "silhouette_score": "Range [-1, 1]: Higher is better. Measures cluster compactness and separation.",
            "davies_bouldin_index": "Range [0, ∞]: Lower is better. Ratio of within to between cluster distances.",
            "calinski_harabasz_index": "Range [0, ∞]: Higher is better. Ratio of between to within cluster dispersion.",
            "inertia_wcss": "Range [0, ∞]: Lower is better. Sum of squared distances to cluster centers.",
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Metrics saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate clustering metrics from trained model")
    parser.add_argument("--model", type=Path, help="Path to saved model (.pkl file)")
    
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
    
    print("=" * 80)
    print("🔍 CLUSTERING METRICS EVALUATION")
    print("=" * 80 + "\n")
    
    # Load model
    pipeline = load_model(args.model)
    
    # Load and prepare data
    sms_df, email_df = load_datasets()
    combined_df = prepare_data(sms_df, email_df)
    
    # Extract features
    X_matrix = extract_features(pipeline, combined_df)
    
    # Get cluster assignments for distribution
    cluster_labels = assign_clusters(pipeline.artifacts.cluster_model, X_matrix)
    get_cluster_distribution(pipeline, cluster_labels)
    
    # Calculate metrics
    metrics = get_clustering_metrics(pipeline, X_matrix)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save results
    save_metrics(metrics, cluster_labels, output_dir, args.model.name)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
