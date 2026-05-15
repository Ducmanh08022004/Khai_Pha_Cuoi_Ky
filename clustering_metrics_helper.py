#!/usr/bin/env python3
"""
Quick helper module to extract clustering metrics from a trained model.

Usage:
    from clustering_metrics_helper import ClusteringMetricsHelper
    
    helper = ClusteringMetricsHelper(pipeline, X_matrix)
    metrics = helper.calculate_all_metrics()
    helper.print_summary()
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.clustering import assign_clusters


class ClusteringMetricsHelper:
    """Helper class to calculate and display clustering metrics."""
    
    def __init__(self, pipeline, X_matrix):
        """
        Initialize the helper.
        
        Args:
            pipeline: Trained FraudDetectionPipeline instance
            X_matrix: Feature matrix (dense or sparse)
        """
        self.pipeline = pipeline
        self.X_matrix = X_matrix
        self.X_dense = self._to_dense(X_matrix)
        self.cluster_labels = assign_clusters(pipeline.artifacts.cluster_model, X_matrix)
        self.cluster_model = pipeline.artifacts.cluster_model
        self.metrics = {}
    
    @staticmethod
    def _to_dense(X):
        """Convert sparse matrix to dense if needed."""
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.asarray(X)
    
    def calculate_silhouette_score(self):
        """Calculate Silhouette Score (range: [-1, 1], higher is better)."""
        try:
            score = silhouette_score(self.X_dense, self.cluster_labels)
            self.metrics['silhouette_score'] = float(score)
            return score
        except Exception as e:
            print(f"⚠️  Error calculating Silhouette Score: {e}")
            self.metrics['silhouette_score'] = None
            return None
    
    def calculate_davies_bouldin_index(self):
        """Calculate Davies-Bouldin Index (range: [0, ∞], lower is better)."""
        try:
            score = davies_bouldin_score(self.X_dense, self.cluster_labels)
            self.metrics['davies_bouldin_index'] = float(score)
            return score
        except Exception as e:
            print(f"⚠️  Error calculating Davies-Bouldin Index: {e}")
            self.metrics['davies_bouldin_index'] = None
            return None
    
    def calculate_calinski_harabasz_index(self):
        """Calculate Calinski-Harabasz Index (range: [0, ∞], higher is better)."""
        try:
            score = calinski_harabasz_score(self.X_dense, self.cluster_labels)
            self.metrics['calinski_harabasz_index'] = float(score)
            return score
        except Exception as e:
            print(f"⚠️  Error calculating Calinski-Harabasz Index: {e}")
            self.metrics['calinski_harabasz_index'] = None
            return None
    
    def get_inertia(self):
        """Get Inertia/WCSS from cluster model (range: [0, ∞], lower is better)."""
        try:
            inertia = self.cluster_model.inertia_
            self.metrics['inertia_wcss'] = float(inertia)
            return inertia
        except Exception as e:
            print(f"⚠️  Error getting Inertia: {e}")
            self.metrics['inertia_wcss'] = None
            return None
    
    def calculate_all_metrics(self):
        """Calculate all clustering metrics at once."""
        print("🔍 Calculating clustering metrics...")
        
        self.calculate_silhouette_score()
        self.calculate_davies_bouldin_index()
        self.calculate_calinski_harabasz_index()
        self.get_inertia()
        
        # Add basic info
        self.metrics['num_clusters'] = int(self.cluster_model.n_clusters)
        self.metrics['num_samples'] = int(len(self.cluster_labels))
        self.metrics['unique_clusters'] = int(len(np.unique(self.cluster_labels)))
        
        return self.metrics
    
    def get_metrics_dict(self):
        """Get metrics as dictionary."""
        return self.metrics.copy()
    
    def get_metrics_summary(self):
        """Get human-readable summary of metrics."""
        summary = []
        summary.append("╔" + "═" * 78 + "╗")
        summary.append("║" + " CLUSTERING METRICS SUMMARY ".center(78) + "║")
        summary.append("╠" + "═" * 78 + "╣")
        
        summary.append(f"║ Model Configuration:".ljust(79) + "║")
        summary.append(f"║   • Clusters (k): {self.metrics.get('num_clusters', 'N/A')} | Samples: {self.metrics.get('num_samples', 'N/A')} | Unique: {self.metrics.get('unique_clusters', 'N/A')}".ljust(79) + "║")
        
        summary.append("║" + "─" * 78 + "║")
        summary.append(f"║ Quality Metrics:".ljust(79) + "║")
        
        # Silhouette
        sil = self.metrics.get('silhouette_score')
        if sil is not None:
            quality = "Excellent" if sil >= 0.5 else ("Good" if sil >= 0.4 else ("Fair" if sil >= 0.25 else "Poor"))
            summary.append(f"║   • Silhouette Score: {sil:.4f} ({quality})".ljust(79) + "║")
            summary.append(f"║     ├─ Range: [-1, 1] | Higher is better".ljust(79) + "║")
            summary.append(f"║     └─ Measures: Cluster compactness & separation".ljust(79) + "║")
        
        # Davies-Bouldin
        db = self.metrics.get('davies_bouldin_index')
        if db is not None:
            quality = "Excellent" if db <= 1.0 else ("Good" if db <= 1.5 else ("Fair" if db <= 2.0 else "Poor"))
            summary.append(f"║   • Davies-Bouldin Index: {db:.4f} ({quality})".ljust(79) + "║")
            summary.append(f"║     ├─ Range: [0, ∞] | Lower is better".ljust(79) + "║")
            summary.append(f"║     └─ Measures: Within vs between cluster distances".ljust(79) + "║")
        
        # Calinski-Harabasz
        ch = self.metrics.get('calinski_harabasz_index')
        if ch is not None:
            quality = "Excellent" if ch >= 1000 else ("Good" if ch >= 500 else ("Fair" if ch >= 100 else "Poor"))
            summary.append(f"║   • Calinski-Harabasz Index: {ch:.4f} ({quality})".ljust(79) + "║")
            summary.append(f"║     ├─ Range: [0, ∞] | Higher is better".ljust(79) + "║")
            summary.append(f"║     └─ Measures: Between vs within cluster dispersion".ljust(79) + "║")
        
        # Inertia
        inertia = self.metrics.get('inertia_wcss')
        if inertia is not None:
            summary.append(f"║   • Inertia (WCSS): {inertia:.4f}".ljust(79) + "║")
            summary.append(f"║     ├─ Range: [0, ∞] | Lower is better".ljust(79) + "║")
            summary.append(f"║     └─ Measures: Sum of squared distances to centers".ljust(79) + "║")
        
        summary.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(summary)
    
    def print_summary(self):
        """Print metrics summary to console."""
        print("\n" + self.get_metrics_summary() + "\n")
    
    def get_cluster_distribution(self):
        """Get cluster size distribution."""
        unique_clusters, counts = np.unique(self.cluster_labels, return_counts=True)
        total_samples = len(self.cluster_labels)
        
        distribution = {}
        for cluster_id, count in zip(unique_clusters, counts):
            percentage = (count / total_samples) * 100
            risk_score = self.pipeline.artifacts.cluster_risk_map.get(int(cluster_id), 0.0)
            distribution[int(cluster_id)] = {
                'samples': int(count),
                'percentage': float(percentage),
                'risk_score': float(risk_score)
            }
        
        return distribution
    
    def print_cluster_distribution(self):
        """Print cluster distribution to console."""
        dist = self.get_cluster_distribution()
        
        print("\n" + "─" * 70)
        print("CLUSTER DISTRIBUTION".center(70))
        print("─" * 70)
        print(f"{'Cluster':>8} | {'Samples':>8} | {'Percentage':>12} | {'Risk Score':>12}")
        print("─" * 70)
        
        for cluster_id in sorted(dist.keys()):
            info = dist[cluster_id]
            print(f"{cluster_id:8d} | {info['samples']:8d} | {info['percentage']:11.2f}% | {info['risk_score']:12.4f}")
        
        print("─" * 70 + "\n")


# Example usage function
def example_usage():
    """Example of how to use this helper class."""
    from src.pipeline import FraudDetectionPipeline
    from src.preprocessing import preprocess_dataframe
    from src.features import transform_feature_bundle
    import pandas as pd
    from pathlib import Path
    
    # Load model
    model_path = Path("models") / "fraud_detection_model.pkl"
    pipeline = FraudDetectionPipeline.load_model(model_path)
    
    # Load data
    data_df = pd.read_csv("dataset/processed/spam_normalized.csv")
    
    # Preprocess and extract features
    prepared_df = preprocess_dataframe(data_df, text_col='text', language='en')
    X_matrix = transform_feature_bundle(pipeline.artifacts.feature_bundle, prepared_df, text_col='clean_text')
    
    # Create helper and calculate metrics
    helper = ClusteringMetricsHelper(pipeline, X_matrix)
    metrics = helper.calculate_all_metrics()
    
    # Display results
    helper.print_summary()
    helper.print_cluster_distribution()
    
    # Access individual metrics
    print(f"Silhouette Score: {metrics['silhouette_score']}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']}")
    print(f"Inertia (WCSS): {metrics['inertia_wcss']}")


if __name__ == "__main__":
    # Run example
    example_usage()
