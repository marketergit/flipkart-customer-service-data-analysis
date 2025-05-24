"""
Utility functions for the Flipkart Customer Support Analysis project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib

def save_model(model, model_name, models_dir='models'):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : sklearn model object
        The trained model to save
    model_name : str
        Name to use for the saved model file
    models_dir : str
        Directory to save models in
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save the model
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_name, models_dir='models'):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_name : str
        Name of the model file to load
    models_dir : str
        Directory where models are saved
    
    Returns:
    --------
    model : sklearn model object
        The loaded model
    """
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    return model

def create_output_dirs():
    """Create necessary output directories for results"""
    dirs = ['models', 'plots', 'results']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_path=None):
    """
    Plot a confusion matrix for model evaluation
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    classes : list or array
        Class names
    model_name : str
        Name of the model for plot title
    save_path : str, optional
        Path to save the plot image
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, save_path=None):
    """
    Plot a ROC curve for model evaluation
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_prob : array-like
        Predicted probabilities for the positive class
    model_name : str
        Name of the model for plot title
    save_path : str, optional
        Path to save the plot image
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    return auc

def get_model_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate various metrics for model evaluation
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for the positive class
    
    Returns:
    --------
    metrics : dict
        Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            # Handle multiclass case or other issues
            metrics['auc'] = None
    
    return metrics
