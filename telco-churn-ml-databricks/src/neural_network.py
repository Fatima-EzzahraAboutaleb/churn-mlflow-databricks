
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')


class NeuralNetworkTrainer:
    """
    Comprehensive Neural Network trainer for binary classification.
    
    Builds and trains Feedforward Neural Networks with various architectures.
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.metrics = {}
        
    def build_model(self, input_dim, hidden_layers=[64, 32], 
                   dropout_rate=0.3, learning_rate=0.001):
        """
        Build a Feedforward Neural Network.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            keras.Model: Compiled model
        """
        if not TENSORFLOW_AVAILABLE:
            print(" TensorFlow not available!")
            return None
        
        model = Sequential(name='Churn_Prediction_NN')
        
        # Input layer + First hidden layer
        model.add(Dense(hidden_layers[0], activation='relu', 
                       input_dim=input_dim, name='input_layer'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Additional hidden layers
        for idx, units in enumerate(hidden_layers[1:], start=2):
            model.add(Dense(units, activation='relu', name=f'hidden_layer_{idx}'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer (sigmoid for binary classification)
        model.add(Dense(1, activation='sigmoid', name='output_layer'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy', 
                             keras.metrics.Precision(name='precision'),
                             keras.metrics.Recall(name='recall'),
                             keras.metrics.AUC(name='auc')])
        
        print("Neural Network Architecture:")
        print("="*60)
        model.summary()
        print("="*60)
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                   epochs=100, batch_size=32, validation_split=0.2,
                   early_stopping=True, patience=10, verbose=1):
        """
        Train the neural network.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            X_val (np.array, optional): Validation features
            y_val (np.array, optional): Validation labels
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training
            validation_split (float): Validation split if X_val not provided
            early_stopping (bool): Whether to use early stopping
            patience (int): Patience for early stopping
            verbose (int): Verbosity mode
            
        Returns:
            keras.callbacks.History: Training history
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("Error: Model not built or TensorFlow not available!")
            return None
        
        print("\n" + "="*60)
        print("Training Neural Network...")
        print("="*60)
        
        # Setup callbacks
        callbacks = []
        
        if early_stopping:
            es = EarlyStopping(monitor='val_loss', patience=patience, 
                             restore_best_weights=True, verbose=1)
            callbacks.append(es)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                     patience=5, min_lr=1e-7, verbose=1)
        callbacks.append(reduce_lr)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split_param = None
        else:
            validation_data = None
            validation_split_param = validation_split
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=validation_split_param,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        print("\n Training Complete!")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test set.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("Error: Model not trained or TensorFlow not available!")
            return None
        
        print("\n" + "="*60)
        print("Evaluating Neural Network on Test Set...")
        print("="*60)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'model_name': 'Neural Network',
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n Neural Network Evaluation:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Precision: {metrics['test_precision']:.4f}")
        print(f"  Recall: {metrics['test_recall']:.4f}")
        print(f"  F1-Score: {metrics['test_f1']:.4f}")
        print(f"  ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {cm}")
        
        self.metrics = metrics
        self.metrics['confusion_matrix'] = cm
        self.metrics['y_pred'] = y_pred
        self.metrics['y_pred_proba'] = y_pred_proba
        
        return metrics
    
    def plot_training_history(self, figsize=(15, 5)):
        """
        Plot training history (loss and metrics).
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.history is None:
            print("Error: No training history available!")
            return None
        
        history_dict = self.history.history
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Loss
        axes[0].plot(history_dict['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[1].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: AUC
        axes[2].plot(history_dict['auc'], label='Training AUC', linewidth=2)
        axes[2].plot(history_dict['val_auc'], label='Validation AUC', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('AUC', fontsize=12)
        axes[2].set_title('Model AUC', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_test, figsize=(8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            y_test (np.array): True labels
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if 'confusion_matrix' not in self.metrics:
            print("Error: Model not evaluated yet!")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   ax=ax, cbar=True)
        
        ax.set_title('Neural Network - Confusion Matrix', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_test, figsize=(8, 6)):
        """
        Plot ROC curve.
        
        Args:
            y_test (np.array): True labels
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if 'y_pred_proba' not in self.metrics:
            print("Error: Model not evaluated yet!")
            return None
        
        y_pred_proba = self.metrics['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = self.metrics['test_roc_auc']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, linewidth=3, label=f'Neural Network (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Neural Network', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print(" TensorFlow not available or No model to save!")
            return
        
        self.model.save(filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not TENSORFLOW_AVAILABLE:
            print("Error: TensorFlow not available!")
            return
        
        self.model = keras.models.load_model(filepath)
        print(f" Model loaded from: {filepath}")


def create_simple_model(input_dim, learning_rate=0.001):
    """
    Create a simple neural network (2 hidden layers).
    
    Args:
        input_dim (int): Number of input features
        learning_rate (float): Learning rate
        
    Returns:
        NeuralNetworkTrainer: Trainer with built model
    """
    trainer = NeuralNetworkTrainer()
    trainer.build_model(input_dim=input_dim, 
                       hidden_layers=[64, 32],
                       dropout_rate=0.3,
                       learning_rate=learning_rate)
    return trainer


def create_deep_model(input_dim, learning_rate=0.001):
    """
    Create a deep neural network (3 hidden layers).
    
    Args:
        input_dim (int): Number of input features
        learning_rate (float): Learning rate
        
    Returns:
        NeuralNetworkTrainer: Trainer with built model
    """
    trainer = NeuralNetworkTrainer()
    trainer.build_model(input_dim=input_dim,
                       hidden_layers=[128, 64, 32],
                       dropout_rate=0.4,
                       learning_rate=learning_rate)
    return trainer


def create_wide_model(input_dim, learning_rate=0.001):
    """
    Create a wide neural network (large hidden layers).
    
    Args:
        input_dim (int): Number of input features
        learning_rate (float): Learning rate
        
    Returns:
        NeuralNetworkTrainer: Trainer with built model
    """
    trainer = NeuralNetworkTrainer()
    trainer.build_model(input_dim=input_dim,
                       hidden_layers=[256, 128],
                       dropout_rate=0.5,
                       learning_rate=learning_rate)
    return trainer


if __name__ == "__main__":
    print("="*60)
    print("Telco Churn Neural Network Module")
    print("="*60)
    print("Module loaded successfully")
    print(f" TensorFlow available: {TENSORFLOW_AVAILABLE}")
    if TENSORFLOW_AVAILABLE:
        print(f" TensorFlow version: {tf.__version__}")
