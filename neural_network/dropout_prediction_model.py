import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import os

# Create neural_network directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def load_and_preprocess_data(cohort_number):
    """Load and preprocess data for a specific cohort."""
    # Load the cohort data
    file_path = f'../cohorts/cohort{cohort_number}.csv'
    data = pd.read_csv(file_path)
    
    # Determine the target column based on cohort
    if cohort_number == 1:
        target_col = 'dropped_after_test_1'
    elif cohort_number == 2:
        target_col = 'dropped_after_test_2'
    elif cohort_number == 3:
        target_col = 'dropped_after_test_3'
    else:  # cohort 4
        target_col = 'final_dropout'
    
    # Get features and target
    # Common features across all cohorts
    common_features = ['External', 'Year', 'session 1', 'session 2', 'test 1']
    
    # Add cohort-specific features
    if cohort_number >= 2:
        common_features.extend(['session 3', 'session 4', 'test 2'])
    if cohort_number >= 3:
        common_features.extend(['session 5', 'test 3'])
    if cohort_number == 4:
        common_features.extend(['session 6', 'ind cw', 'group cw', 'final grade', 
                              'fourm Q', 'fourm A', 'office hour visits'])
    
    # Select features and target
    X = data[common_features]
    y = data[target_col]
    
    return X, y

def build_model(input_dim):
    """Build a neural network model for dropout prediction."""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_model(model, X_test, y_test, cohort_number):
    """Evaluate the model and generate performance metrics and plots."""
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nCohort {cohort_number} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Cohort {cohort_number} - Confusion Matrix')
    plt.savefig(f'plots/cohort{cohort_number}_confusion_matrix.png')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cohort {cohort_number} - ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'plots/cohort{cohort_number}_roc_curve.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc
    }

def train_and_evaluate_cohort(cohort_number):
    """Train and evaluate a model for a specific cohort."""
    print(f"\n===== Cohort {cohort_number} =====")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(cohort_number)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    model = build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'Cohort {cohort_number} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'Cohort {cohort_number} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/cohort{cohort_number}_training_history.png')
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test, cohort_number)
    
    # Save model
    model.save(f'models/cohort{cohort_number}_model.h5')
    
    return metrics, model

def analyze_feature_importance(model, feature_names, cohort_number):
    """Analyze and visualize feature importance."""
    # Get weights from the first layer
    weights = model.layers[0].get_weights()[0]
    
    # Calculate absolute weight sum for each feature
    importance = np.sum(np.abs(weights), axis=1)
    
    # Create a feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.xticks(rotation=90)
    plt.title(f'Cohort {cohort_number} - Feature Importance')
    plt.tight_layout()
    plt.savefig(f'plots/cohort{cohort_number}_feature_importance.png')
    
    return feature_importance

def compare_models(metrics_list):
    """Compare performance of models across cohorts."""
    cohorts = [1, 2, 3, 4]
    metrics_df = pd.DataFrame({
        'Cohort': cohorts,
        'Accuracy': [m['accuracy'] for m in metrics_list],
        'Precision': [m['precision'] for m in metrics_list],
        'Recall': [m['recall'] for m in metrics_list],
        'F1 Score': [m['f1'] for m in metrics_list],
        'AUC': [m['auc'] for m in metrics_list]
    })
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        plt.bar([f"Cohort {c}" for c in cohorts], metrics_df[metric])
        plt.title(metric)
        plt.ylim(0, 1)
        
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    
    return metrics_df

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    all_metrics = []
    
    # Train and evaluate models for each cohort
    for cohort in range(1, 5):
        metrics, model = train_and_evaluate_cohort(cohort)
        all_metrics.append(metrics)
        
        # Get feature names and analyze importance
        X, _ = load_and_preprocess_data(cohort)
        feature_names = X.columns.tolist()
        analyze_feature_importance(model, feature_names, cohort)
    
    # Compare model performance
    comparison = compare_models(all_metrics)
    print("\nModel Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv('models/model_comparison.csv', index=False)

if __name__ == "__main__":
    main() 