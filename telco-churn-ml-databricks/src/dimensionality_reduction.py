
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class DimensionalityReducer:
    """
    Comprehensive dimensionality reduction for visualization and analysis.
    """
    
    def __init__(self):
        self.pca_model = None
        self.tsne_model = None
        self.nmf_model = None
        self.results = {}
        
    def fit_pca(self, X, n_components=2):
        """
        Fit PCA (Principal Component Analysis).
        
        Args:
            X (np.array): Feature matrix
            n_components (int): Number of components
            
        Returns:
            tuple: (transformed_data, model)
        """
        print("\n" + "="*60)
        print(f"Applying PCA with {n_components} components...")
        print("="*60)
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"✓ PCA complete!")
        print(f"  Explained variance ratio: {explained_var}")
        print(f"  Cumulative explained variance: {cumulative_var[-1]:.4f}")
        
        self.pca_model = pca
        self.results['pca'] = {
            'transformed': X_pca,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        }
        
        return X_pca, pca
    
    def fit_tsne(self, X, n_components=2, perplexity=30, n_iter=1000):
        """
        Fit t-SNE (t-Distributed Stochastic Neighbor Embedding).
        
        Args:
            X (np.array): Feature matrix
            n_components (int): Number of components
            perplexity (float): Perplexity parameter (5-50)
            n_iter (int): Number of iterations
            
        Returns:
            tuple: (transformed_data, model)
        """
        print("\n" + "="*60)
        print(f"Applying t-SNE with {n_components} components...")
        print(f"  Perplexity: {perplexity}, Iterations: {n_iter}")
        print("="*60)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   n_iter=n_iter, random_state=42, verbose=0)
        X_tsne = tsne.fit_transform(X)
        
        print(f"✓ t-SNE complete!")
        print(f"  Final KL divergence: {tsne.kl_divergence_:.4f}")
        
        self.tsne_model = tsne
        self.results['tsne'] = {
            'transformed': X_tsne,
            'kl_divergence': tsne.kl_divergence_
        }
        
        return X_tsne, tsne
    
    def fit_nmf(self, X, n_components=2):
        """
        Fit NMF (Non-negative Matrix Factorization).
        Note: NMF requires non-negative input.
        
        Args:
            X (np.array): Feature matrix (must be non-negative)
            n_components (int): Number of components
            
        Returns:
            tuple: (transformed_data, model)
        """
        print("\n" + "="*60)
        print(f"Applying NMF with {n_components} components...")
        print("="*60)
        
        # Ensure non-negative data
        if X.min() < 0:
            print("  ⚠ Data contains negative values. Applying MinMax scaling...")
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
        X_nmf = nmf.fit_transform(X_scaled)
        
        print(f"✓ NMF complete!")
        print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")
        
        self.nmf_model = nmf
        self.results['nmf'] = {
            'transformed': X_nmf,
            'reconstruction_error': nmf.reconstruction_err_
        }
        
        return X_nmf, nmf
    
    def plot_pca_variance(self, figsize=(12, 5)):
        """
        Plot PCA explained variance.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.pca_model is None:
            print(" PCA not fitted yet!")
            return None
        
        explained_var = self.pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Variance per component
        axes[0].bar(range(1, len(explained_var) + 1), explained_var, 
                   color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Variance Explained by Each Component', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Cumulative variance
        axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                    marker='o', linewidth=2, markersize=8, color='darkorange')
        axes[1].axhline(y=0.95, color='r', linestyle='--', 
                       label='95% variance', linewidth=2)
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Variance Explained', 
                         fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_projection(self, X_transformed, y, method_name, figsize=(10, 8)):
        """
        Plot 2D projection colored by target variable.
        
        Args:
            X_transformed (np.array): 2D transformed data
            y (np.array): Target labels
            method_name (str): Name of the method (PCA, t-SNE, NMF)
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                           c=y, cmap='coolwarm', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
        ax.set_title(f'{method_name} - 2D Projection (Colored by Churn)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
        cbar.set_label('Churn', fontsize=11)
        cbar.ax.set_yticklabels(['No', 'Yes'])
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_all_projections(self, y, figsize=(18, 6)):
        """
        Plot all 2D projections side by side.
        
        Args:
            y (np.array): Target labels
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        methods = []
        if 'pca' in self.results:
            methods.append(('PCA', self.results['pca']['transformed']))
        if 'tsne' in self.results:
            methods.append(('t-SNE', self.results['tsne']['transformed']))
        if 'nmf' in self.results:
            methods.append(('NMF', self.results['nmf']['transformed']))
        
        if not methods:
            print(" No dimensionality reduction methods applied yet!")
            return None
        
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, X_transformed) in enumerate(methods):
            scatter = axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1],
                                       c=y, cmap='coolwarm', alpha=0.6, s=30, 
                                       edgecolors='k', linewidth=0.5)
            
            axes[idx].set_xlabel(f'{method_name} Component 1', fontsize=11)
            axes[idx].set_ylabel(f'{method_name} Component 2', fontsize=11)
            axes[idx].set_title(f'{method_name} Projection', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add colorbar to last subplot
            if idx == n_methods - 1:
                cbar = plt.colorbar(scatter, ax=axes[idx], ticks=[0, 1])
                cbar.set_label('Churn', fontsize=10)
                cbar.ax.set_yticklabels(['No', 'Yes'])
        
        plt.suptitle('Dimensionality Reduction - All Methods', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def analyze_pca_components(self, feature_names, n_components=5):
        """
        Analyze which features contribute most to each principal component.
        
        Args:
            feature_names (list): List of original feature names
            n_components (int): Number of components to analyze
            
        Returns:
            pd.DataFrame: Component analysis dataframe
        """
        if self.pca_model is None:
            print("PCA not fitted yet!")
            return None
        
        # Get component loadings
        components = self.pca_model.components_[:n_components]
        
        # Create dataframe
        df_components = pd.DataFrame(
            components.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
        
        print("\n" + "="*80)
        print("PCA COMPONENT ANALYSIS - Top Contributing Features")
        print("="*80)
        
        for i in range(n_components):
            pc_name = f'PC{i+1}'
            top_features = df_components[pc_name].abs().sort_values(ascending=False).head(5)
            
            print(f"\n{pc_name} (Variance: {self.pca_model.explained_variance_ratio_[i]:.4f}):")
            for feature, loading in top_features.items():
                sign = '+' if df_components.loc[feature, pc_name] > 0 else '-'
                print(f"  {sign} {feature:30s}: {abs(loading):.4f}")
        
        print("="*80 + "\n")
        
        return df_components
    
    def plot_pca_loadings(self, feature_names, component_indices=[0, 1], 
                          top_n=10, figsize=(12, 6)):
        """
        Plot feature loadings for specified principal components.
        
        Args:
            feature_names (list): List of feature names
            component_indices (list): Indices of components to plot
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.pca_model is None:
            print("PCA not fitted yet!")
            return None
        
        n_plots = len(component_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for idx, comp_idx in enumerate(component_indices):
            loadings = self.pca_model.components_[comp_idx]
            
            # Get top N features
            top_indices = np.argsort(np.abs(loadings))[-top_n:]
            top_loadings = loadings[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            # Create colors based on sign
            colors = ['red' if x < 0 else 'blue' for x in top_loadings]
            
            axes[idx].barh(top_features, top_loadings, color=colors, alpha=0.7)
            axes[idx].set_xlabel('Loading', fontsize=11)
            axes[idx].set_title(f'PC{comp_idx+1} - Top {top_n} Features', 
                              fontsize=12, fontweight='bold')
            axes[idx].axvline(x=0, color='k', linestyle='--', linewidth=1)
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig


def find_optimal_n_components(X, variance_threshold=0.95):
    """
    Find optimal number of PCA components for given variance threshold.
    
    Args:
        X (np.array): Feature matrix
        variance_threshold (float): Desired cumulative variance
        
    Returns:
        int: Optimal number of components
    """
    pca = PCA(random_state=42)
    pca.fit(X)
    
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_var >= variance_threshold) + 1
    
    print(f"\n To explain {variance_threshold*100}% variance: {n_components} components needed")
    print(f"  (from total of {X.shape[1]} features)")
    
    return n_components


if __name__ == "__main__":
    print("="*60)
    print("Telco Churn Dimensionality Reduction Module")
    print("="*60)
    print("Module loaded successfully")
