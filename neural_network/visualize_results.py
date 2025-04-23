import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

# Import the data loading function from the main script
from dropout_prediction_model import load_and_preprocess_data

def load_model_and_data(cohort_number, use_best_model=False):
    """Load model and data for a specific cohort."""
    # Load data
    X, y = load_and_preprocess_data(cohort_number)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load model
    model_path = f'hyperparameter_tuning/cohort{cohort_number}_best_model.h5' if use_best_model else f'models/cohort{cohort_number}_model.h5'
    
    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load model from {model_path}")
        return None, None, None, None, None
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test

def plot_roc_curves_all_cohorts(use_best_model=False):
    """Plot ROC curves for all cohorts on the same figure."""
    plt.figure(figsize=(10, 8))
    
    for cohort in range(1, 5):
        model, _, X_test, _, y_test = load_model_and_data(cohort, use_best_model)
        
        if model is None:
            continue
        
        # Predict probabilities
        y_pred_prob = model.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'Cohort {cohort} (AUC = {roc_auc:.4f})')
    
    # Add reference line
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Cohorts')
    plt.legend(loc='lower right')
    
    # Save plot
    model_type = 'best_models' if use_best_model else 'default_models'
    os.makedirs('visualization', exist_ok=True)
    plt.savefig(f'visualization/roc_curves_all_cohorts_{model_type}.png')
    plt.close()

def plot_precision_recall_curves_all_cohorts(use_best_model=False):
    """Plot precision-recall curves for all cohorts on the same figure."""
    plt.figure(figsize=(10, 8))
    
    for cohort in range(1, 5):
        model, _, X_test, _, y_test = load_model_and_data(cohort, use_best_model)
        
        if model is None:
            continue
        
        # Predict probabilities
        y_pred_prob = model.predict(X_test)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, label=f'Cohort {cohort} (AUC = {pr_auc:.4f})')
    
    # Add labels and legend
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Cohorts')
    plt.legend(loc='lower left')
    
    # Save plot
    model_type = 'best_models' if use_best_model else 'default_models'
    plt.savefig(f'visualization/precision_recall_curves_all_cohorts_{model_type}.png')
    plt.close()

def plot_confusion_matrices(use_best_model=False):
    """Plot confusion matrices for all cohorts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, cohort in enumerate(range(1, 5)):
        model, _, X_test, _, y_test = load_model_and_data(cohort, use_best_model)
        
        if model is None:
            continue
        
        # Predict classes
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_title(f'Cohort {cohort} - Confusion Matrix')
    
    plt.tight_layout()
    
    # Save plot
    model_type = 'best_models' if use_best_model else 'default_models'
    plt.savefig(f'visualization/confusion_matrices_{model_type}.png')
    plt.close()

def plot_feature_importance_comparison(use_best_model=False):
    """Compare feature importance across cohorts."""
    # First, get feature names for each cohort
    feature_names = []
    for cohort in range(1, 5):
        X, _ = load_and_preprocess_data(cohort)
        feature_names.append(X.columns.tolist())
    
    # Find common features across all cohorts
    common_features = set(feature_names[0])
    for features in feature_names[1:]:
        common_features = common_features.intersection(set(features))
    
    common_features = list(common_features)
    
    # Create a figure for common features
    plt.figure(figsize=(12, 8))
    
    # For each cohort, extract importance values for common features
    for cohort in range(1, 5):
        model, _, _, _, _ = load_model_and_data(cohort, use_best_model)
        
        if model is None:
            continue
        
        # Get weights from the first layer
        weights = model.layers[0].get_weights()[0]
        
        # Calculate absolute weight sum for each feature
        all_importance = np.sum(np.abs(weights), axis=1)
        
        # Map to feature names
        X, _ = load_and_preprocess_data(cohort)
        all_features = X.columns.tolist()
        
        # Extract importance for common features
        importance = []
        for feature in common_features:
            idx = all_features.index(feature)
            importance.append(all_importance[idx])
        
        # Normalize importance values to [0, 1] range
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        # Plot bar for this cohort
        x = np.arange(len(common_features))
        width = 0.2  # Width of bars
        plt.bar(x + (cohort - 2.5) * width, importance, width, label=f'Cohort {cohort}')
    
    # Add labels and legend
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance Comparison Across Cohorts')
    plt.xticks(x, common_features, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    model_type = 'best_models' if use_best_model else 'default_models'
    plt.savefig(f'visualization/feature_importance_comparison_{model_type}.png')
    plt.close()

def plot_prediction_distribution(use_best_model=False):
    """Plot distribution of prediction probabilities for each cohort."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, cohort in enumerate(range(1, 5)):
        model, _, X_test, _, y_test = load_model_and_data(cohort, use_best_model)
        
        if model is None:
            continue
        
        # Predict probabilities
        y_pred_prob = model.predict(X_test).flatten()
        
        # Split predictions by actual class
        dropout_probs = y_pred_prob[y_test == 1]
        non_dropout_probs = y_pred_prob[y_test == 0]
        
        # Plot histograms
        axes[i].hist(non_dropout_probs, bins=20, alpha=0.5, label='Non-Dropout')
        axes[i].hist(dropout_probs, bins=20, alpha=0.5, label='Dropout')
        axes[i].set_xlabel('Predicted Probability of Dropout')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Cohort {cohort} - Prediction Distribution')
        axes[i].legend()
    
    plt.tight_layout()
    
    # Save plot
    model_type = 'best_models' if use_best_model else 'default_models'
    plt.savefig(f'visualization/prediction_distribution_{model_type}.png')
    plt.close()

def main():
    """Main function to generate all visualizations."""
    # Create visualization directory if it doesn't exist
    os.makedirs('visualization', exist_ok=True)
    
    # Generate visualizations for default models
    print("Generating visualizations for default models...")
    plot_roc_curves_all_cohorts(use_best_model=False)
    plot_precision_recall_curves_all_cohorts(use_best_model=False)
    plot_confusion_matrices(use_best_model=False)
    plot_feature_importance_comparison(use_best_model=False)
    plot_prediction_distribution(use_best_model=False)
    
    # Generate visualizations for best models (if they exist)
    if os.path.exists('hyperparameter_tuning'):
        print("Generating visualizations for best models...")
        plot_roc_curves_all_cohorts(use_best_model=True)
        plot_precision_recall_curves_all_cohorts(use_best_model=True)
        plot_confusion_matrices(use_best_model=True)
        plot_feature_importance_comparison(use_best_model=True)
        plot_prediction_distribution(use_best_model=True)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 