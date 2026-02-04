

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')


def create_metrics_comparison_table(metrics_list):
    """
    Create a comprehensive comparison table of all models.
    
    Args:
        metrics_list (list): List of metrics dictionaries from different models
        
    Returns:
        pd.DataFrame: Styled comparison dataframe
    """
    df = pd.DataFrame(metrics_list)
    
    # Sort by test accuracy
    df = df.sort_values('test_accuracy', ascending=False).reset_index(drop=True)
    
    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df


def print_model_comparison(metrics_list):
    """
    Print a beautiful comparison table.
    
    Args:
        metrics_list (list): List of metrics dictionaries
    """
    df = create_metrics_comparison_table(metrics_list)
    
    print("\n" + "="*120)
    print("FINAL MODEL COMPARISON - ALL ALGORITHMS")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    # Print best model
    best_model = df.iloc[0]
    print(f"\n BEST MODEL: {best_model['model_name']}")
    print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"   F1-Score: {best_model['test_f1']:.4f}")
    print(f"   ROC-AUC: {best_model['test_roc_auc']:.4f}")
    print("="*120 + "\n")


def plot_model_comparison_bars(metrics_list, figsize=(14, 8)):
    """
    Plot bar charts comparing models across different metrics.
    
    Args:
        metrics_list (list): List of metrics dictionaries
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    df = create_metrics_comparison_table(metrics_list)
    
    metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        bars = ax.barh(df['model_name'], df[metric], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel(name, fontsize=11)
        ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_heatmap(metrics_list, figsize=(12, 8)):
    """
    Plot heatmap of all metrics for all models.
    
    Args:
        metrics_list (list): List of metrics dictionaries
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    df = create_metrics_comparison_table(metrics_list)
    
    # Select only metric columns
    metric_cols = ['test_accuracy', 'test_precision', 'test_recall', 
                   'test_f1', 'test_roc_auc']
    df_metrics = df[['model_name'] + metric_cols].set_index('model_name')
    
    # Rename columns for display
    df_metrics.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='YlGnBu', 
               cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix_grid(predictions_dict, figsize=(16, 12)):
    """
    Plot grid of confusion matrices for multiple models.
    
    Args:
        predictions_dict (dict): Dictionary of {model_name: (y_true, y_pred)}
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    n_models = len(predictions_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   ax=axes[idx], cbar=True)
        
        axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_roc_curves_comparison(roc_data_dict, figsize=(12, 8)):
    """
    Plot ROC curves for multiple models on the same plot.
    
    Args:
        roc_data_dict (dict): Dictionary of {model_name: (y_true, y_proba)}
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(roc_data_dict)))
    
    for idx, (model_name, (y_true, y_proba)) in enumerate(roc_data_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2.5, color=colors[idx],
               label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - All Models Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 8)):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not hasattr(model, 'feature_importances_'):
        print(" Model does not have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Create dataframe
    df_importance = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(df_importance)))
    
    bars = ax.barh(df_importance['feature'], df_importance['importance'], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_classification_report(y_true, y_pred, model_name):
    """
    Generate and print detailed classification report.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        model_name (str): Name of the model
    """
    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT - {model_name}")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    print("="*60 + "\n")


def export_results_to_excel(metrics_list, filepath):
    """
    Export model comparison results to Excel file.
    
    Args:
        metrics_list (list): List of metrics dictionaries
        filepath (str): Path to save Excel file
    """
    df = create_metrics_comparison_table(metrics_list)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Model Comparison', index=False)
        
        # Style the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Model Comparison']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f" Results exported to: {filepath}")


def calculate_model_ranking(metrics_list):
    """
    Calculate overall ranking of models based on multiple metrics.
    
    Args:
        metrics_list (list): List of metrics dictionaries
        
    Returns:
        pd.DataFrame: Ranking dataframe
    """
    df = pd.DataFrame(metrics_list)
    
    # Define weights for each metric
    weights = {
        'test_accuracy': 0.25,
        'test_precision': 0.20,
        'test_recall': 0.20,
        'test_f1': 0.25,
        'test_roc_auc': 0.10
    }
    
    # Calculate weighted score
    df['weighted_score'] = sum(df[metric] * weight for metric, weight in weights.items())
    
    # Sort and rank
    df = df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df[['Rank', 'model_name', 'weighted_score', 'test_accuracy', 'test_f1', 'test_roc_auc']]


if __name__ == "__main__":
    print("="*60)
    print("Telco Churn Evaluation Module")
    print("="*60)
    print(" Module loaded successfully")
