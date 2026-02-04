
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SupervisedModelTrainer:
    """
    Comprehensive trainer for supervised learning models.
    
    Manages training, evaluation, and comparison of multiple algorithms.
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.roc_curves = {}
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, 
                                  C=1.0, max_iter=1000, random_state=42):
        """
        Train Logistic Regression classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            C: Regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
        
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test, 
                                      model_name='Logistic Regression')
        
        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = metrics
        
        return model, metrics
    
    def train_svm(self, X_train, y_train, X_test, y_test,
                  kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """
        Train Support Vector Machine classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print(f"Training SVM (kernel={kernel})...")
        print("="*60)
        
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state, 
                   probability=True)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name=f'SVM ({kernel})')
        
        self.models[f'svm_{kernel}'] = model
        self.metrics[f'svm_{kernel}'] = metrics
        
        return model, metrics
    
    def train_knn(self, X_train, y_train, X_test, y_test,
                  n_neighbors=5, weights='uniform', metric='minkowski'):
        """
        Train K-Nearest Neighbors classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_neighbors: Number of neighbors
            weights: Weight function ('uniform', 'distance')
            metric: Distance metric
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print(f"Training KNN (k={n_neighbors})...")
        print("="*60)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, 
                                    metric=metric)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name=f'KNN (k={n_neighbors})')
        
        self.models['knn'] = model
        self.metrics['knn'] = metrics
        
        return model, metrics
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test,
                           max_depth=None, min_samples_split=2, random_state=42):
        """
        Train Decision Tree classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print("Training Decision Tree...")
        print("="*60)
        
        model = DecisionTreeClassifier(max_depth=max_depth, 
                                      min_samples_split=min_samples_split,
                                      random_state=random_state)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name='Decision Tree')
        
        self.models['decision_tree'] = model
        self.metrics['decision_tree'] = metrics
        
        return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test,
                           n_estimators=100, max_depth=None, random_state=42):
        """
        Train Random Forest classifier (Bagging ensemble).
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print(f"Training Random Forest (n_estimators={n_estimators})...")
        print("="*60)
        
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                      max_depth=max_depth,
                                      random_state=random_state)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name='Random Forest')
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = metrics
        
        return model, metrics
    
    def train_adaboost(self, X_train, y_train, X_test, y_test,
                      n_estimators=50, learning_rate=1.0, random_state=42):
        """
        Train AdaBoost classifier (Boosting ensemble).
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_estimators: Number of estimators
            learning_rate: Learning rate
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print(f"Training AdaBoost (n_estimators={n_estimators})...")
        print("="*60)
        
        model = AdaBoostClassifier(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  random_state=random_state)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name='AdaBoost')
        
        self.models['adaboost'] = model
        self.metrics['adaboost'] = metrics
        
        return model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test,
                               n_estimators=100, learning_rate=0.1, 
                               max_depth=3, random_state=42):
        """
        Train Gradient Boosting classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of trees
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics)
        """
        print("\n" + "="*60)
        print(f"Training Gradient Boosting (n_estimators={n_estimators})...")
        print("="*60)
        
        model = GradientBoostingClassifier(n_estimators=n_estimators,
                                          learning_rate=learning_rate,
                                          max_depth=max_depth,
                                          random_state=random_state)
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name='Gradient Boosting')
        
        self.models['gradient_boosting'] = model
        self.metrics['gradient_boosting'] = metrics
        
        return model, metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test,
                     n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Train XGBoost classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
            
        Returns:
            tuple: (model, metrics) or (None, None) if XGBoost not available
        """
        if not XGBOOST_AVAILABLE:
            print("⚠ XGBoost not available. Skipping...")
            return None, None
        
        print("\n" + "="*60)
        print(f"Training XGBoost (n_estimators={n_estimators})...")
        print("="*60)
        
        model = xgb.XGBClassifier(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 max_depth=max_depth,
                                 random_state=random_state,
                                 eval_metric='logloss')
        model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test,
                                      model_name='XGBoost')
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = metrics
        
        return model, metrics
    
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_name: Name of the model
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = model.decision_function(X_train)
            y_test_proba = model.decision_function(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba),
        }
        
        # Store predictions and probabilities
        self.predictions[model_name] = {
            'y_test': y_test,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        self.roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr, 
                                       'auc': metrics['test_roc_auc']}
        
        # Print results
        print(f"\n✓ {model_name} Training Complete!")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.4f} | Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test Precision: {metrics['test_precision']:.4f} | Recall: {metrics['test_recall']:.4f}")
        print(f"  Test F1-Score: {metrics['test_f1']:.4f} | ROC-AUC: {metrics['test_roc_auc']:.4f}")
        
        return metrics
    
    def compare_all_models(self):
        """
        Create comparison table of all trained models.
        
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_key, metrics in self.metrics.items():
            comparison_data.append(metrics)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Reorder columns
        cols_order = ['model_name', 'test_accuracy', 'test_precision', 'test_recall', 
                     'test_f1', 'test_roc_auc', 'train_accuracy']
        df_comparison = df_comparison[cols_order]
        
        # Sort by test accuracy
        df_comparison = df_comparison.sort_values('test_accuracy', ascending=False)
        
        print("\n" + "="*100)
        print("MODEL COMPARISON - SUPERVISED LEARNING")
        print("="*100)
        print(df_comparison.to_string(index=False))
        print("="*100)
        
        return df_comparison
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """
        Plot ROC curves for all models on the same plot.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.roc_curves)))
        
        for idx, (model_name, roc_data) in enumerate(self.roc_curves.items()):
            ax.plot(roc_data['fpr'], roc_data['tpr'], 
                   label=f"{model_name} (AUC = {roc_data['auc']:.3f})",
                   linewidth=2.5, color=colors[idx])
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title('ROC Curves - All Models Comparison', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self, figsize=(16, 12)):
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        n_models = len(self.predictions)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, pred_data) in enumerate(self.predictions.items()):
            cm = confusion_matrix(pred_data['y_test'], pred_data['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11)
            axes[idx].set_xlabel('Predicted Label', fontsize=11)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("="*60)
    print("Telco Churn Supervised Learning Module")
    print("="*60)
    print("Module loaded successfully")
    print(f" XGBoost available: {XGBOOST_AVAILABLE}")
