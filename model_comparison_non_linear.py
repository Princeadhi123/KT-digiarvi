"""
Knowledge Tracing Model Comparison with Non-linear Algorithms

This script compares the performance of various non-linear machine learning models
for predicting student performance in knowledge tracing tasks.
"""

import os
import warnings
from typing import Tuple, Dict, Any, List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'output_dir': 'model_outputs_non_linear',
    'test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5
}

# Set up output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the knowledge tracing data.
    
    Args:
        filepath: Path to the CSV file containing the data
        
    Returns:
        Preprocessed pandas DataFrame
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df['sex'] = df['sex'].fillna('Unknown')
    df['mean_perception'] = df['mean_perception'].fillna(df['mean_perception'].mean())
    
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, list, list]:
    """
    Prepare features and target variable for modeling.
    
    Args:
        df: Input DataFrame with raw features
        
    Returns:
        Tuple containing:
            - Features (X)
            - Target (y)
            - Fitted preprocessor
            - List of numeric feature names
            - List of categorical feature names
    """
    # Define feature columns
    feature_columns = [
        'category', 'order', 'grade', 'sex', 'mean_perception',
        'time_spent', 'total_attempts', 'cumulative_passes', 'pass_rate'
    ]
    X = df[feature_columns]
    y = df['score']
    
    # Define feature types
    numeric_features = [
        'order', 'mean_perception', 'time_spent',
        'total_attempts', 'cumulative_passes', 'pass_rate'
    ]
    categorical_features = ['category', 'sex']
    
    # Create and fit preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )
    
    return X, y, preprocessor, numeric_features, categorical_features

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Evaluate model performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary containing evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== {model_name} ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    top_n: int = 15
) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model for plot title
        top_n: Number of top features to display
    """
    if not hasattr(model, 'feature_importances_'):
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importances - {model_name}")
    plt.bar(
        range(len(indices)),
        importances[indices],
        align='center'
    )
    
    plt.xticks(
        range(len(indices)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha='right'
    )
    
    plt.tight_layout()
    output_path = os.path.join(
        CONFIG['output_dir'],
        f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    )
    plt.savefig(output_path)
    plt.close()

def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    model_name: str,
    model: Any,
    feature_names: List[str]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train and evaluate a model.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        preprocessor: Fitted preprocessor
        model_name: Name of the model
        model: Model instance
        feature_names: List of feature names
        
    Returns:
        Tuple of (fitted_model, metrics_dict)
    """
    print(f"\n=== Training {model_name} ===")
    
    # Create and train pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Evaluate
    print("\nTraining Set Performance:")
    train_metrics = evaluate_model(y_train, y_pred_train, f"{model_name} (Train)")
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, f"{model_name} (Test)")
    
    # Plot feature importance
    if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        plot_feature_importance(
            pipeline.named_steps['regressor'],
            feature_names,
            model_name
        )
    
    return pipeline, test_metrics

def get_model_instances() -> Dict[str, Any]:
    """
    Get model instances with their configurations.
    
    Returns:
        Dictionary of model names to model instances
    """
    return {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=CONFIG['n_estimators'],
            learning_rate=CONFIG['learning_rate'],
            max_depth=CONFIG['max_depth'],
            random_state=CONFIG['random_state'],
            n_jobs=CONFIG['n_jobs']
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=CONFIG['n_estimators'],
            learning_rate=CONFIG['learning_rate'],
            max_depth=CONFIG['max_depth'],
            random_state=CONFIG['random_state'],
            n_jobs=CONFIG['n_jobs']
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=CONFIG['n_estimators'],
            max_depth=CONFIG['max_depth'],
            random_state=CONFIG['random_state'],
            n_jobs=CONFIG['n_jobs']
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=CONFIG['n_estimators'],
            learning_rate=CONFIG['learning_rate'],
            max_depth=CONFIG['max_depth'],
            random_state=CONFIG['random_state']
        )
    }

def plot_model_comparison(metrics_list: List[Dict[str, Any]]) -> None:
    """
    Create comparison plots for model performance metrics.
    
    Args:
        metrics_list: List of metric dictionaries from evaluate_model
    """
    df_metrics = pd.DataFrame(metrics_list)
    
    # Plot R² comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='R2', data=df_metrics)
    plt.title('Model Comparison - R² Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'model_comparison_r2.png'))
    plt.close()
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='RMSE', data=df_metrics)
    plt.title('Model Comparison - RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'model_comparison_rmse.png'))
    plt.close()

def save_models(models: Dict[str, Any]) -> None:
    """
    Save trained models to disk.
    
    Args:
        models: Dictionary of model names to model instances
    """
    for name, model in models.items():
        filename = os.path.join(CONFIG['output_dir'], f"{name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, filename)

def main():
    """Main execution function."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('preprocessed_kt_data.csv')
    
    # Prepare features and target
    X, y, preprocessor, numeric_features, categorical_features = prepare_features(df)
    
    # Get feature names after one-hot encoding
    preprocessor.fit(X)
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + ohe_feature_names.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train and evaluate models
    models = {}
    metrics_list = []
    
    for name, model in get_model_instances().items():
        model_pipeline, metrics = train_model(
            X_train, X_test, y_train, y_test,
            preprocessor, name, model, all_feature_names
        )
        models[name] = model_pipeline
        metrics_list.append(metrics)
    
    # Save models and generate comparison plots
    save_models(models)
    plot_model_comparison(metrics_list)
    
    print("\n=== Model Training and Evaluation Completed ===")
    print(f"Models and visualizations have been saved in the '{CONFIG['output_dir']}' directory.")

if __name__ == "__main__":
    main()
