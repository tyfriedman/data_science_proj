import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
import os
import json
import glob
from tensorflow.keras.models import load_model
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Only show errors, not warnings

# Import the data loading function from the main script
from dropout_prediction_model import load_and_preprocess_data

def find_available_models():
    """Find all available models in the models/ and hyperparameter_tuning/ directories."""
    models = {
        'default': {},
        'tuned': {}
    }
    
    # Check for default models
    os.makedirs('models', exist_ok=True)
    for cohort in range(1, 5):
        # Check for both possible extensions
        h5_path = f'models/cohort{cohort}_model.h5'
        keras_path = f'models/cohort{cohort}_model.keras'
        
        if os.path.exists(h5_path):
            models['default'][cohort] = h5_path
        elif os.path.exists(keras_path):
            models['default'][cohort] = keras_path
    
    # Check for tuned models
    os.makedirs('hyperparameter_tuning', exist_ok=True)
    for cohort in range(1, 5):
        # Check for both possible extensions
        h5_path = f'hyperparameter_tuning/cohort{cohort}_best_model.h5'
        keras_path = f'hyperparameter_tuning/cohort{cohort}_best_model.keras'
        
        if os.path.exists(h5_path):
            models['tuned'][cohort] = h5_path
        elif os.path.exists(keras_path):
            models['tuned'][cohort] = keras_path
    
    return models

def get_feature_info(cohort, model_type):
    """Get feature information for a specific cohort."""
    if model_type == 'tuned':
        info_path = f'hyperparameter_tuning/cohort{cohort}_feature_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
    
    # If no feature info file or not a tuned model, return basic info
    X, _ = load_and_preprocess_data(cohort)
    return {
        'feature_count': X.shape[1],
        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    }

def get_model_feature_names(model_path):
    """Extract original feature names used to train the model from feature info files."""
    try:
        # Determine model type and cohort from path
        if 'hyperparameter_tuning' in model_path:
            # Extract cohort number from the path
            cohort = int(model_path.split('cohort')[1].split('_')[0])
            info_path = f'hyperparameter_tuning/cohort{cohort}_feature_info.json'
            
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    feature_info = json.load(f)
                if 'feature_names' in feature_info:
                    return feature_info['feature_names']
    except Exception as e:
        print(f"Error reading model feature names: {str(e)}")
    
    return None

def select_matching_features(model_path, X, expected_input_shape):
    """Select features to match the expected input shape for the model."""
    if not hasattr(X, 'columns'):
        return None
        
    data_features = X.columns.tolist()
    
    # Try to get the original feature names
    original_features = get_model_feature_names(model_path)
    
    # Case 1: We have original feature names that match the expected shape
    if original_features and len(original_features) == expected_input_shape:
        # Check how many original features exist in our current dataset
        common_features = [f for f in original_features if f in data_features]
        
        # If we have an exact match, use it
        if len(common_features) == expected_input_shape:
            print(f"Using exact original features ({len(common_features)})")
            return X[common_features]
            
        # If we have most of the features, use what we have
        elif len(common_features) >= expected_input_shape * 0.8:  # At least 80% match
            print(f"Using {len(common_features)}/{expected_input_shape} original features")
            return X[common_features].iloc[:, :expected_input_shape]  # Take only as many as needed
    
    # Case 2: Use a cohort-specific selection approach
    if expected_input_shape == 8:
        # Most likely the cohort 1 model (8 features)
        # Common features for cohort 1 are usually the first 8
        cohort1_common = ['External', 'Year', 'session 1', 'session 2', 'test 1', 
                          'forum_Q_phase1', 'forum_A_phase1', 'office_hours_phase1']
        
        # Check if these specific features exist in our dataset
        common_features = [f for f in cohort1_common if f in data_features]
        if len(common_features) == 8:
            print(f"Using cohort 1 standard features")
            return X[common_features]
        
        # Fallback to first 8 features
        print(f"Fallback: Using first 8 features")
        return X.iloc[:, :8]
        
    elif expected_input_shape == 10:
        # Likely a cohort 2 model - try standard features if they exist
        print(f"Fallback: Using first 10 features")
        return X.iloc[:, :10]
        
    elif expected_input_shape == 13:
        # Likely a cohort 3 model
        print(f"Fallback: Using first 13 features")
        return X.iloc[:, :13]
        
    else:
        # For any other input shape, use first N features
        print(f"Fallback: Using first {expected_input_shape} features")
        if X.shape[1] >= expected_input_shape:
            return X.iloc[:, :expected_input_shape]
    
    # If we can't figure out how to select features, return None
    return None

def get_model_input_shape(model):
    """Safely extract the input shape from a Keras model."""
    try:
        # Try direct model.input_shape access
        if hasattr(model, 'input_shape'):
            if model.input_shape is not None:
                # Most straightforward case - the model has an input_shape attribute
                input_shape = model.input_shape
                if isinstance(input_shape, tuple) and len(input_shape) > 1:
                    return input_shape[1]  # Return feature dimension
        
        # Try alternative approaches
        if hasattr(model, '_layers') and len(model._layers) > 0:
            first_layer = model._layers[0]
            if hasattr(first_layer, 'input_shape'):
                input_shape = first_layer.input_shape
                if input_shape is not None and isinstance(input_shape, tuple) and len(input_shape) > 1:
                    return input_shape[1]
        
        # Try accessing weights directly
        if hasattr(model, 'get_weights') and len(model.get_weights()) > 0:
            # First layer weights may have shape (input_dim, output_dim)
            first_weights = model.get_weights()[0]
            if len(first_weights.shape) == 2:
                return first_weights.shape[0]  # First dimension of weights
        
        # If we still can't determine, try looking at individual layers
        for layer in model.layers:
            if 'Dense' in str(layer.__class__):
                if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                    weights = layer.get_weights()[0]
                    if len(weights.shape) == 2:
                        # For the first Dense layer, weights shape is (input_dim, units)
                        return weights.shape[0]
            
            # Try to check input_shape on this layer
            if hasattr(layer, 'input_shape'):
                shape = layer.input_shape
                if shape is not None and isinstance(shape, tuple) and len(shape) > 1:
                    return shape[1]
        
        return None
    
    except Exception as e:
        print(f"Error determining input shape: {str(e)}")
        return None

def get_model_weights_shape(model):
    """Extract the weights shape of the first layer as a fallback for input dimension."""
    try:
        if hasattr(model, 'get_weights') and len(model.get_weights()) > 0:
            weights = model.get_weights()
            # Typically first layer weights will have shape (input_features, output_units)
            if len(weights) > 0 and hasattr(weights[0], 'shape') and len(weights[0].shape) > 0:
                return weights[0].shape
        return None
    except Exception as e:
        print(f"Error getting weights shape: {str(e)}")
        return None

def get_cohort_specific_input_shape(model_path, cohort):
    """Get expected input shape based on cohort number if model doesn't provide it."""
    # Different cohorts have different known feature counts
    cohort_to_features = {
        1: 8,   # Cohort 1 typically has 8 features
        2: 11,  # Cohort 2 typically has 11 features
        3: 12,  # Cohort 3 typically has 12 features
        4: 14   # Cohort 4 typically has 14 features
    }
    
    return cohort_to_features.get(cohort, None)

def load_model_and_data(cohort, model_path):
    """Load model and prepare data for a specific cohort."""
    try:
        # Load the model
        model = load_model(model_path)
        print(f"Loaded model: {model_path}")
        
        # Determine model type
        model_type = 'default' if 'models/' in model_path else 'tuned'
        
        # Get feature info
        feature_info = get_feature_info(cohort, model_type)
        
        # Load and preprocess data
        X, y = load_and_preprocess_data(cohort)
        
        # Try to get the expected input shape from the model
        expected_input_shape = get_model_input_shape(model)
        
        # If we couldn't determine the input shape directly, try to check weights
        if expected_input_shape is None:
            weights_shape = get_model_weights_shape(model)
            if weights_shape is not None and len(weights_shape) > 0:
                expected_input_shape = weights_shape[0]
                print(f"Determined input shape from weights: {expected_input_shape}")
        
        # If we still couldn't determine, try cohort-specific fallbacks
        if expected_input_shape is None:
            expected_input_shape = get_cohort_specific_input_shape(model_path, cohort)
            if expected_input_shape is not None:
                print(f"Using cohort-specific input shape: {expected_input_shape} for cohort {cohort}")
            else:
                print(f"Could not determine input shape for model. Assuming it matches the data: {X.shape[1]}")
                expected_input_shape = X.shape[1]
        else:
            print(f"Model expects input shape: {expected_input_shape}")
        
        # Check if input dimensions match what the model expects
        if hasattr(X, 'columns') and X.shape[1] != expected_input_shape:
            print(f"Input shape mismatch for cohort {cohort}: model expects {expected_input_shape} features but data has {X.shape[1]}")
            
            # Try to select appropriate features
            X_selected = select_matching_features(model_path, X, expected_input_shape)
            
            if X_selected is not None:
                X = X_selected
            else:
                print(f"ERROR: Cannot match feature dimensions for cohort {cohort}")
                return None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Final check that dimensions match the model's expectations
        if X_test_scaled.shape[1] != expected_input_shape:
            print(f"ERROR: Feature count still mismatched after preprocessing: got {X_test_scaled.shape[1]}, expected {expected_input_shape}")
            return None, None, None, None
            
        return model, X_test_scaled, y_test, feature_info
        
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None, None, None, None

def visualize_roc_curves(models_dict, model_type):
    """Visualize ROC curves for all available models."""
    plt.figure(figsize=(10, 8))
    
    has_models = False
    
    for cohort, model_path in models_dict.items():
        model, X_test, y_test, _ = load_model_and_data(cohort, model_path)
        
        if model is None or X_test is None or y_test is None:
            continue
        
        has_models = True
        
        # Predict probabilities
        try:
            y_pred_prob = model.predict(X_test)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'Cohort {cohort} (AUC = {roc_auc:.4f})')
        except Exception as e:
            print(f"Error generating ROC curve for cohort {cohort}: {str(e)}")
    
    if has_models:
        # Add reference line
        plt.plot([0, 1], [0, 1], 'k--')
        
        # Add labels and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {model_type.capitalize()} Models')
        plt.legend(loc='lower right')
        
        # Save plot
        os.makedirs('visualization', exist_ok=True)
        plt.savefig(f'visualization/roc_curves_{model_type}.png')
        print(f"Saved ROC curves for {model_type} models")
    else:
        print(f"No valid models found for ROC curve visualization ({model_type})")
    
    plt.close()

def visualize_precision_recall_curves(models_dict, model_type):
    """Visualize precision-recall curves for all available models."""
    plt.figure(figsize=(10, 8))
    
    has_models = False
    
    for cohort, model_path in models_dict.items():
        model, X_test, y_test, _ = load_model_and_data(cohort, model_path)
        
        if model is None or X_test is None or y_test is None:
            continue
        
        has_models = True
        
        # Predict probabilities
        try:
            y_pred_prob = model.predict(X_test)
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
            pr_auc = auc(recall, precision)
            
            # Plot precision-recall curve
            plt.plot(recall, precision, label=f'Cohort {cohort} (AUC = {pr_auc:.4f})')
        except Exception as e:
            print(f"Error generating precision-recall curve for cohort {cohort}: {str(e)}")
    
    if has_models:
        # Add labels and legend
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves for {model_type.capitalize()} Models')
        plt.legend(loc='lower left')
        
        # Save plot
        os.makedirs('visualization', exist_ok=True)
        plt.savefig(f'visualization/precision_recall_curves_{model_type}.png')
        print(f"Saved precision-recall curves for {model_type} models")
    else:
        print(f"No valid models found for precision-recall curve visualization ({model_type})")
    
    plt.close()

def visualize_confusion_matrices(models_dict, model_type):
    """Visualize confusion matrices for all available models."""
    # Count valid models first
    valid_models = []
    for cohort, model_path in models_dict.items():
        model, X_test, y_test, _ = load_model_and_data(cohort, model_path)
        if model is not None and X_test is not None and y_test is not None:
            valid_models.append((cohort, model, X_test, y_test))
    
    if not valid_models:
        print(f"No valid models found for confusion matrix visualization ({model_type})")
        return
    
    # Create a grid based on number of valid models
    n_models = len(valid_models)
    rows = int(np.ceil(n_models / 2))
    cols = min(2, n_models)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (cohort, model, X_test, y_test) in enumerate(valid_models):
        try:
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
        except Exception as e:
            print(f"Error generating confusion matrix for cohort {cohort}: {str(e)}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i].set_title(f'Cohort {cohort} - Error')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('visualization', exist_ok=True)
    plt.savefig(f'visualization/confusion_matrices_{model_type}.png')
    print(f"Saved confusion matrices for {model_type} models")
    plt.close()

def visualize_feature_importance(models_dict, model_type):
    """Visualize feature importance for all available models."""
    # Get all models and their feature info
    model_features = {}
    
    for cohort, model_path in models_dict.items():
        model, _, _, feature_info = load_model_and_data(cohort, model_path)
        
        if model is None or feature_info is None:
            continue
            
        try:
            # Get weights from the first layer
            weights = model.layers[0].get_weights()[0]
            
            # Calculate absolute weight sum for each feature
            importance = np.sum(np.abs(weights), axis=1)
            
            # Store importance and feature names
            if 'feature_names' in feature_info and len(importance) == len(feature_info['feature_names']):
                model_features[cohort] = {
                    'importance': importance,
                    'feature_names': feature_info['feature_names']
                }
        except Exception as e:
            print(f"Error calculating feature importance for cohort {cohort}: {str(e)}")
    
    if not model_features:
        print(f"No valid models found for feature importance visualization ({model_type})")
        return
        
    # Individual feature importance plots
    for cohort, data in model_features.items():
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame for importance
        feature_df = pd.DataFrame({
            'Feature': data['feature_names'],
            'Importance': data['importance']
        })
        
        # Sort by importance
        feature_df = feature_df.sort_values('Importance', ascending=False)
        
        # Plot
        sns.barplot(x='Importance', y='Feature', data=feature_df)
        plt.title(f'Cohort {cohort} - Feature Importance ({model_type.capitalize()} Model)')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('visualization', exist_ok=True)
        plt.savefig(f'visualization/feature_importance_cohort{cohort}_{model_type}.png')
        plt.close()
    
    print(f"Saved feature importance plots for {model_type} models")

def visualize_prediction_distribution(models_dict, model_type):
    """Visualize prediction distribution for all available models."""
    # Count valid models first
    valid_models = []
    for cohort, model_path in models_dict.items():
        model, X_test, y_test, _ = load_model_and_data(cohort, model_path)
        if model is not None and X_test is not None and y_test is not None:
            valid_models.append((cohort, model, X_test, y_test))
    
    if not valid_models:
        print(f"No valid models found for prediction distribution visualization ({model_type})")
        return
    
    # Create a grid based on number of valid models
    n_models = len(valid_models)
    rows = int(np.ceil(n_models / 2))
    cols = min(2, n_models)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (cohort, model, X_test, y_test) in enumerate(valid_models):
        try:
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
        except Exception as e:
            print(f"Error generating prediction distribution for cohort {cohort}: {str(e)}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i].set_title(f'Cohort {cohort} - Error')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('visualization', exist_ok=True)
    plt.savefig(f'visualization/prediction_distribution_{model_type}.png')
    print(f"Saved prediction distribution plots for {model_type} models")
    plt.close()

def main():
    """Main function to run all visualizations."""
    print("Finding available models...")
    available_models = find_available_models()
    
    print(f"Found {len(available_models['default'])} default models and {len(available_models['tuned'])} tuned models")
    
    # Create visualization directory
    os.makedirs('visualization', exist_ok=True)
    
    # Visualize default models if any exist
    if available_models['default']:
        print("\nGenerating visualizations for default models...")
        visualize_roc_curves(available_models['default'], 'default')
        visualize_precision_recall_curves(available_models['default'], 'default')
        visualize_confusion_matrices(available_models['default'], 'default')
        visualize_feature_importance(available_models['default'], 'default')
        visualize_prediction_distribution(available_models['default'], 'default')
    else:
        print("No default models found to visualize")
    
    # Visualize tuned models if any exist
    if available_models['tuned']:
        print("\nGenerating visualizations for tuned models...")
        visualize_roc_curves(available_models['tuned'], 'tuned')
        visualize_precision_recall_curves(available_models['tuned'], 'tuned')
        visualize_confusion_matrices(available_models['tuned'], 'tuned')
        visualize_feature_importance(available_models['tuned'], 'tuned')
        visualize_prediction_distribution(available_models['tuned'], 'tuned')
    else:
        print("No tuned models found to visualize")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 