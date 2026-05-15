#!/usr/bin/env python3
"""
t-SNE Visualization Helper Module

Provides utilities to visualize clusters using t-SNE dimensionality reduction.
Can be used standalone or imported as a module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

from src.clustering import assign_clusters


class TSNEVisualizationHelper:
    """Helper class for t-SNE cluster visualization."""
    
    def __init__(self, X_matrix, cluster_labels, pipeline=None):
        """
        Initialize t-SNE visualization helper.
        
        Args:
            X_matrix: Feature matrix (dense or sparse)
            cluster_labels: Cluster assignments for each sample
            pipeline: FraudDetectionPipeline instance (optional, for cluster info)
        """
        self.X_matrix = X_matrix
        self.cluster_labels = cluster_labels
        self.pipeline = pipeline
        self.X_tsne = None
        self.unique_clusters = np.unique(cluster_labels)
    
    @staticmethod
    def _to_dense(X):
        """Convert sparse matrix to dense if needed."""
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.asarray(X)
    
    def fit_tsne(self, perplexity=30, max_iter=1000, random_state=42, verbose=1):
        """
        Apply t-SNE dimensionality reduction.
        
        Args:
            perplexity: t-SNE perplexity parameter (5-50, typically 30)
            max_iter: Number of iterations
            random_state: Random seed
            verbose: Verbosity level (0, 1)
        
        Returns:
            X_tsne: 2D t-SNE embedding
        """
        print("🔍 Applying t-SNE dimensionality reduction...")
        print(f"   Perplexity: {perplexity} | Iterations: {max_iter}\n")
        
        X_dense = self._to_dense(self.X_matrix)
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose
        )
        
        self.X_tsne = tsne.fit_transform(X_dense)
        print(f"\n   ✓ t-SNE embedding shape: {self.X_tsne.shape}\n")
        
        return self.X_tsne
    
    def plot_clusters(self, title="t-SNE Visualization of Fraud Clusters", figsize=(14, 10), dpi=100):
        """
        Create t-SNE cluster visualization.
        
        Args:
            title: Plot title
            figsize: Figure size (width, height)
            dpi: Resolution
        
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if self.X_tsne is None:
            raise ValueError("t-SNE has not been fitted yet. Call fit_tsne() first.")
        
        print("📊 Creating t-SNE visualization...")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        n_clusters = len(self.unique_clusters)
        
        # Create colormap
        if n_clusters <= 10:
            colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_clusters))
        elif n_clusters <= 20:
            colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_clusters))
        else:
            colors = cm.get_cmap('hsv')(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for idx, cluster_id in enumerate(sorted(self.unique_clusters)):
            mask = self.cluster_labels == cluster_id
            scatter = ax.scatter(
                self.X_tsne[mask, 0],
                self.X_tsne[mask, 1],
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
        
        print(f"   ✓ Visualization created\n")
        
        return fig, ax
    
    def plot_clusters_with_density(self, title="t-SNE Visualization with Density", figsize=(14, 10), dpi=100):
        """
        Create t-SNE visualization with density heatmap.
        
        Args:
            title: Plot title
            figsize: Figure size
            dpi: Resolution
        
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        if self.X_tsne is None:
            raise ValueError("t-SNE has not been fitted yet. Call fit_tsne() first.")
        
        print("📊 Creating t-SNE visualization with density...")
        
        from scipy.stats import gaussian_kde
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Calculate density
        xy = np.vstack([self.X_tsne[:, 0], self.X_tsne[:, 1]])
        z = gaussian_kde(xy)(xy)
        
        # Sort by density
        idx = z.argsort()
        x, y, z = self.X_tsne[idx, 0], self.X_tsne[idx, 1], z[idx]
        
        scatter = ax.scatter(
            x, y,
            c=z,
            s=50,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.3
        )
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Density', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        print(f"   ✓ Visualization created\n")
        
        return fig, ax
    
    def save_plot(self, fig, output_path):
        """Save plot to file."""
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Plot saved to: {output_path}")
        plt.close(fig)
    
    def get_cluster_stats(self):
        """Get cluster statistics."""
        unique_clusters, counts = np.unique(self.cluster_labels, return_counts=True)
        total_samples = len(self.cluster_labels)
        
        stats = {}
        for cluster_id, count in zip(unique_clusters, counts):
            percentage = (count / total_samples) * 100
            risk_score = None
            if self.pipeline:
                risk_score = self.pipeline.artifacts.cluster_risk_map.get(int(cluster_id), 0.0)
            
            stats[int(cluster_id)] = {
                'samples': int(count),
                'percentage': float(percentage),
                'risk_score': float(risk_score) if risk_score is not None else None
            }
        
        return stats
    
    def print_cluster_stats(self):
        """Print cluster statistics."""
        stats = self.get_cluster_stats()
        
        print("📊 Cluster Statistics:")
        print("─" * 70)
        print(f"{'Cluster':>8} | {'Samples':>8} | {'Percentage':>12} | {'Risk Score':>12}")
        print("─" * 70)
        
        for cluster_id in sorted(stats.keys()):
            info = stats[cluster_id]
            risk_str = f"{info['risk_score']:.4f}" if info['risk_score'] is not None else "N/A"
            print(f"{cluster_id:8d} | {info['samples']:8d} | {info['percentage']:11.2f}% | {risk_str:>12}")
        
        print("─" * 70)


# Example usage
def example_usage():
    """Example of how to use the helper class."""
    from src.pipeline import FraudDetectionPipeline
    from src.preprocessing import preprocess_dataframe
    from src.features import transform_feature_bundle
    from pathlib import Path
    
    # Load model and data
    model_path = Path("models") / "fraud_detection_model.pkl"
    pipeline = FraudDetectionPipeline.load_model(model_path)
    
    data_df = pd.read_csv("dataset/processed/spam_normalized.csv")
    prepared_df = preprocess_dataframe(data_df, text_col='text', language='en')
    X_matrix = transform_feature_bundle(pipeline.artifacts.feature_bundle, prepared_df, text_col='clean_text')
    
    # Get cluster labels
    cluster_labels = assign_clusters(pipeline.artifacts.cluster_model, X_matrix)
    
    # Create visualization helper
    helper = TSNEVisualizationHelper(X_matrix, cluster_labels, pipeline)
    
    # Fit t-SNE
    helper.fit_tsne(perplexity=30, max_iter=1000)
    
    # Create and display visualization
    fig, ax = helper.plot_clusters()
    plt.show()
    
    # Or with density
    fig, ax = helper.plot_clusters_with_density()
    plt.show()
    
    # Print statistics
    helper.print_cluster_stats()
    
    # Save plot
    helper.save_plot(fig, "results/tsne_clusters.png")


if __name__ == "__main__":
    example_usage()
