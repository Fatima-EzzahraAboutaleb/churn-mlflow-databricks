

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def find_optimal_k(self, X, k_range=(2, 11), method='elbow'):
        
        scores = {'k': [], 'inertia': [], 'silhouette': []}
        
        for k in range(k_range[0], k_range[1]):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            scores['k'].append(k)
            scores['inertia'].append(kmeans.inertia_)
            scores['silhouette'].append(silhouette_score(X, labels))
        
        print(f"✓ Optimal k analysis complete for k={k_range[0]} to {k_range[1]-1}")
        return scores
    
    def plot_elbow_silhouette(self, scores):
        """
        Plot Elbow curve and Silhouette scores.
        
        Args:
            scores (dict): Dictionary from find_optimal_k()
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(scores['k'], scores['inertia'], marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(scores['k'], scores['silhouette'], marker='s', 
                     linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score for Optimal k', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Mark the best silhouette score
        best_k_idx = np.argmax(scores['silhouette'])
        best_k = scores['k'][best_k_idx]
        best_score = scores['silhouette'][best_k_idx]
        axes[1].scatter([best_k], [best_score], color='red', s=200, 
                       zorder=5, marker='*', label=f'Best k={best_k}')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def fit_kmeans(self, X, n_clusters=3, random_state=42):
        """
        Fit K-Means clustering algorithm.
        
        Args:
            X (np.array): Feature matrix
            n_clusters (int): Number of clusters
            random_state (int): Random seed
            
        Returns:
            tuple: (model, labels, metrics)
        """
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        
        metrics = self._calculate_metrics(X, labels)
        metrics['n_clusters'] = n_clusters
        metrics['inertia'] = model.inertia_
        
        self.models['kmeans'] = model
        self.results['kmeans'] = {'labels': labels, 'metrics': metrics}
        
        print(f"✓ K-Means fitted with {n_clusters} clusters")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
        
        return model, labels, metrics
    
    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """
        Fit DBSCAN clustering algorithm.
        
        Args:
            X (np.array): Feature matrix
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in a neighborhood
            
        Returns:
            tuple: (model, labels, metrics)
        """
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate metrics only for non-noise points
        if n_clusters > 1:
            mask = labels != -1
            metrics = self._calculate_metrics(X[mask], labels[mask])
        else:
            metrics = {
                'silhouette_score': -1,
                'davies_bouldin': -1,
                'calinski_harabasz': -1
            }
        
        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = n_noise
        metrics['noise_ratio'] = n_noise / len(labels)
        
        self.models['dbscan'] = model
        self.results['dbscan'] = {'labels': labels, 'metrics': metrics}
        
        print(f"✓ DBSCAN fitted: {n_clusters} clusters, {n_noise} noise points ({metrics['noise_ratio']:.2%})")
        if n_clusters > 1:
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        
        return model, labels, metrics
    
    def fit_hierarchical(self, X, n_clusters=3, linkage_method='ward'):
        """
        Fit Hierarchical (Agglomerative) clustering.
        
        Args:
            X (np.array): Feature matrix
            n_clusters (int): Number of clusters
            linkage_method (str): Linkage method ('ward', 'complete', 'average')
            
        Returns:
            tuple: (model, labels, metrics)
        """
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)
        
        metrics = self._calculate_metrics(X, labels)
        metrics['n_clusters'] = n_clusters
        metrics['linkage'] = linkage_method
        
        self.models['hierarchical'] = model
        self.results['hierarchical'] = {'labels': labels, 'metrics': metrics}
        
        print(f"✓ Hierarchical Clustering fitted with {n_clusters} clusters ({linkage_method} linkage)")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        
        return model, labels, metrics
    
    def _calculate_metrics(self, X, labels):
        """
        Calculate clustering evaluation metrics.
        
        Args:
            X (np.array): Feature matrix
            labels (np.array): Cluster labels
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # Only calculate if we have more than 1 cluster
        if len(set(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        else:
            metrics['silhouette_score'] = -1
            metrics['davies_bouldin'] = -1
            metrics['calinski_harabasz'] = -1
        
        return metrics
    
    def compare_with_ground_truth(self, clustering_labels, true_labels):
        """
        Compare clustering results with ground truth (Churn labels).
        
        Args:
            clustering_labels (np.array): Predicted cluster labels
            true_labels (np.array): True labels (Churn)
            
        Returns:
            dict: Comparison metrics
        """
        metrics = {
            'adjusted_rand_index': adjusted_rand_score(true_labels, clustering_labels),
            'normalized_mutual_info': normalized_mutual_info_score(true_labels, clustering_labels)
        }
        
        print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
        
        return metrics
    
    def plot_dendrogram(self, X, sample_size=1000, figsize=(12, 6)):
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            X (np.array): Feature matrix
            sample_size (int): Number of samples to use (for performance)
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Calculate linkage
        Z = linkage(X_sample, method='ward')
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=30)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def compare_all_algorithms(self, y_true=None):
        """
        Create comparison table of all clustering algorithms.
        
        Args:
            y_true (np.array, optional): True labels for comparison
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for algo_name, result in self.results.items():
            metrics = result['metrics'].copy()
            metrics['Algorithm'] = algo_name.upper()
            
            # Add comparison with ground truth if provided
            if y_true is not None:
                gt_metrics = self.compare_with_ground_truth(result['labels'], y_true)
                metrics.update(gt_metrics)
            
            comparison_data.append(metrics)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Reorder columns
        cols_order = ['Algorithm', 'n_clusters', 'silhouette_score', 'davies_bouldin', 
                     'calinski_harabasz']
        if y_true is not None:
            cols_order.extend(['adjusted_rand_index', 'normalized_mutual_info'])
        
        # Keep only existing columns
        cols_order = [col for col in cols_order if col in df_comparison.columns]
        df_comparison = df_comparison[cols_order]
        
        print("\n" + "="*80)
        print("CLUSTERING ALGORITHMS COMPARISON")
        print("="*80)
        print(df_comparison.to_string(index=False))
        print("="*80)
        
        return df_comparison
    
    def plot_cluster_distributions(self, labels_dict, y_true, figsize=(15, 5)):
        """
        Plot distribution of true labels within each cluster for all algorithms.
        
        Args:
            labels_dict (dict): Dictionary of {algorithm_name: labels}
            y_true (np.array): True labels
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        n_algorithms = len(labels_dict)
        fig, axes = plt.subplots(1, n_algorithms, figsize=figsize)
        
        if n_algorithms == 1:
            axes = [axes]
        
        for idx, (algo_name, labels) in enumerate(labels_dict.items()):
            # Create contingency table
            df_temp = pd.DataFrame({'Cluster': labels, 'Churn': y_true})
            contingency = pd.crosstab(df_temp['Cluster'], df_temp['Churn'], normalize='index') * 100
            
            # Plot
            contingency.plot(kind='bar', stacked=True, ax=axes[idx], 
                           color=['#2ecc71', '#e74c3c'])
            axes[idx].set_title(f'{algo_name.upper()}\nChurn Distribution per Cluster', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Cluster', fontsize=11)
            axes[idx].set_ylabel('Percentage (%)', fontsize=11)
            axes[idx].legend(title='Churn', labels=['No', 'Yes'])
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("="*60)
    print("Telco Churn Clustering Module")
    print("="*60)
    print("✓ Module loaded successfully")
