import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import os
import json

# Import the data loading function from the main script
from dropout_prediction_model import load_and_preprocess_data

def build_model_with_params(input_dim, params):
    """Build a neural network model with the specified hyperparameters."""
    model = Sequential()
    
    # Input layer
    model.add(Dense(
        params['first_layer_size'], 
        activation=params['activation'],
        input_dim=input_dim
    ))
    
    # Add dropout if specified
    if params['dropout_rate'] > 0:
        model.add(Dropout(params['dropout_rate']))
    
    # Add additional hidden layers if specified
    if params['num_hidden_layers'] > 1:
        for i in range(params['num_hidden_layers'] - 1):
            # Decrease layer size by half for each additional layer
            layer_size = params['first_layer_size'] // (2 ** (i + 1))
            if layer_size < 8:  # Minimum layer size
                layer_size = 8
            model.add(Dense(layer_size, activation=params['activation']))
            if params['dropout_rate'] > 0:
                model.add(Dropout(params['dropout_rate'] / 2))  # Less dropout in deeper layers
    
    # Output layer (sigmoid for binary classification)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model with specified optimizer
    optimizer = None
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_with_params(X_train, X_test, y_train, y_test, params):
    """Train and evaluate a model with the specified hyperparameters."""
    # Build model
    model = build_model_with_params(X_train.shape[1], params)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'model': model,
        'history': history,
        'accuracy': accuracy,
        'f1': f1,
        'val_loss': min(history.history['val_loss']),
        'epochs_trained': len(history.history['loss'])
    }

def grid_search(X, y, param_grid, cohort_number):
    """Perform grid search over the parameter grid."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize results
    results = []
    best_f1 = 0
    best_model = None
    best_params = None
    
    # Total number of combinations
    total_combinations = (
        len(param_grid['first_layer_size']) *
        len(param_grid['num_hidden_layers']) *
        len(param_grid['activation']) *
        len(param_grid['dropout_rate']) *
        len(param_grid['optimizer']) *
        len(param_grid['learning_rate']) *
        len(param_grid['batch_size']) *
        len(param_grid['epochs'])
    )
    
    print(f"Cohort {cohort_number} - Performing grid search with {total_combinations} combinations")
    
    # Counter to track progress
    counter = 0
    
    # Iterate over all parameter combinations
    for first_layer_size in param_grid['first_layer_size']:
        for num_hidden_layers in param_grid['num_hidden_layers']:
            for activation in param_grid['activation']:
                for dropout_rate in param_grid['dropout_rate']:
                    for optimizer in param_grid['optimizer']:
                        for learning_rate in param_grid['learning_rate']:
                            for batch_size in param_grid['batch_size']:
                                for epochs in param_grid['epochs']:
                                    # Create parameter dictionary
                                    params = {
                                        'first_layer_size': first_layer_size,
                                        'num_hidden_layers': num_hidden_layers,
                                        'activation': activation,
                                        'dropout_rate': dropout_rate,
                                        'optimizer': optimizer,
                                        'learning_rate': learning_rate,
                                        'batch_size': batch_size,
                                        'epochs': epochs
                                    }
                                    
                                    # Train and evaluate model
                                    result = train_and_evaluate_with_params(
                                        X_train_scaled, X_test_scaled, y_train, y_test, params
                                    )
                                    
                                    # Update results
                                    result_dict = {
                                        **params,
                                        'accuracy': result['accuracy'],
                                        'f1': result['f1'],
                                        'val_loss': result['val_loss'],
                                        'epochs_trained': result['epochs_trained']
                                    }
                                    results.append(result_dict)
                                    
                                    # Update best model if necessary
                                    if result['f1'] > best_f1:
                                        best_f1 = result['f1']
                                        best_model = result['model']
                                        best_params = params
                                    
                                    # Update counter and print progress
                                    counter += 1
                                    if counter % 10 == 0 or counter == total_combinations:
                                        print(f"Completed {counter}/{total_combinations} combinations")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('hyperparameter_tuning', exist_ok=True)
    results_df.to_csv(f'hyperparameter_tuning/cohort{cohort_number}_results.csv', index=False)
    
    # Save best parameters
    with open(f'hyperparameter_tuning/cohort{cohort_number}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Save best model
    best_model.save(f'hyperparameter_tuning/cohort{cohort_number}_best_model.h5')
    
    # Plot F1 Score vs Parameters
    for param in param_grid.keys():
        if len(param_grid[param]) > 1:  # Only plot if parameter was varied
            plt.figure(figsize=(10, 6))
            # Group by parameter and calculate mean F1 score
            grouped = results_df.groupby(param)['f1'].mean().reset_index()
            plt.bar(grouped[param].astype(str), grouped['f1'])
            plt.title(f'Cohort {cohort_number} - Mean F1 Score vs {param}')
            plt.xlabel(param)
            plt.ylabel('Mean F1 Score')
            plt.savefig(f'hyperparameter_tuning/cohort{cohort_number}_{param}_vs_f1.png')
            plt.close()
    
    return results_df, best_params, best_model

def main():
    # Define parameter grid
    param_grid = {
        'first_layer_size': [32, 64, 128],
        'num_hidden_layers': [1, 2, 3],
        'activation': ['relu', 'tanh'],
        'dropout_rate': [0, 0.2, 0.4],
        'optimizer': ['adam', 'rmsprop'],
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'epochs': [50]
    }
    
    # Train and evaluate models for each cohort
    best_params_all = {}
    
    for cohort in range(1, 5):
        print(f"\n===== Tuning Cohort {cohort} =====")
        
        # Load and preprocess data
        X, y = load_and_preprocess_data(cohort)
        
        # Perform grid search
        _, best_params, _ = grid_search(X, y, param_grid, cohort)
        
        # Store best parameters
        best_params_all[f'cohort{cohort}'] = best_params
    
    # Save all best parameters
    with open('hyperparameter_tuning/best_params_all.json', 'w') as f:
        json.dump(best_params_all, f, indent=4)
    
    print("\nHyperparameter tuning complete. Best parameters saved to 'hyperparameter_tuning/best_params_all.json'")

if __name__ == "__main__":
    main() 