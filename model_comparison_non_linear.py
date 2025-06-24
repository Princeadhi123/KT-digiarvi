import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'model_outputs'
os.makedirs(output_dir, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_and_preprocess_data(filepath):
    """Load and preprocess the data."""
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df['sex'] = df['sex'].fillna('Unknown')
    df['mean_perception'] = df['mean_perception'].fillna(df['mean_perception'].mean())
    
    return df

def prepare_features(df):
    """Prepare features and target variable."""
    # Define features and target
    X = df[['category', 'order', 'grade', 'sex', 'mean_perception', 
            'time_spent', 'total_attempts', 'cumulative_passes', 'pass_rate']]
    y = df['score']
    
    # Define preprocessing for different column types
    numeric_features = ['order', 'mean_perception', 'time_spent', 'total_attempts', 
                      'cumulative_passes', 'pass_rate']
    categorical_features = ['category', 'sex']
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate and print model performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== {model_name} Performance ===")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(indices[:15])), importances[indices][:15], align='center')
        plt.xticks(range(len(indices[:15])), [feature_names[i] for i in indices][:15], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.png"))
        plt.close()

def train_random_forest(X_train, X_test, y_train, y_test, preprocessor, feature_names):
    """Train and evaluate Random Forest model."""
    print("\n=== Training Random Forest Regressor ===")
    
    # Create pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train model
    rf_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf_pipeline.predict(X_train)
    y_pred_test = rf_pipeline.predict(X_test)
    
    # Evaluate
    print("\nTraining Set Performance:")
    train_metrics = evaluate_model(y_train, y_pred_train, "Random Forest (Train)")
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Random Forest (Test)")
    
    # Plot feature importance
    if hasattr(rf_pipeline.named_steps['regressor'], 'feature_importances_'):
        plot_feature_importance(
            rf_pipeline.named_steps['regressor'],
            feature_names,
            "Random Forest"
        )
    
    return rf_pipeline, test_metrics

def train_gradient_boosting(X_train, X_test, y_train, y_test, preprocessor, feature_names):
    """Train and evaluate Gradient Boosting model."""
    print("\n=== Training Gradient Boosting Regressor ===")
    
    # Create pipeline
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train model
    gb_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = gb_pipeline.predict(X_train)
    y_pred_test = gb_pipeline.predict(X_test)
    
    # Evaluate
    print("\nTraining Set Performance:")
    train_metrics = evaluate_model(y_train, y_pred_train, "Gradient Boosting (Train)")
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Gradient Boosting (Test)")
    
    # Plot feature importance
    if hasattr(gb_pipeline.named_steps['regressor'], 'feature_importances_'):
        plot_feature_importance(
            gb_pipeline.named_steps['regressor'],
            feature_names,
            "Gradient Boosting"
        )
    
    return gb_pipeline, test_metrics

def plot_model_comparison(metrics_list):
    """Create a bar plot comparing model performances."""
    df_metrics = pd.DataFrame(metrics_list)
    
    # Plot R² comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='R2', data=df_metrics)
    plt.title('Model Comparison - R² Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_r2.png'))
    plt.close()
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='RMSE', data=df_metrics)
    plt.title('Model Comparison - RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_rmse.png'))
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('preprocessed_kt_data.csv')
    
    # Prepare features and target
    X, y, preprocessor = prepare_features(df)
    
    # Get feature names after one-hot encoding
    preprocessor.fit(X)
    numeric_features = ['order', 'mean_perception', 'time_spent', 'total_attempts', 
                       'cumulative_passes', 'pass_rate']
    categorical_features = ['category', 'sex']
    
    # Get one-hot encoded feature names
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + ohe_feature_names.tolist()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train and evaluate models
    metrics_list = []
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(
        X_train, X_test, y_train, y_test, preprocessor, all_feature_names
    )
    metrics_list.append(rf_metrics)
    
    # Save the model
    joblib.dump(rf_model, os.path.join(output_dir, 'random_forest_model.pkl'))
    
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(
        X_train, X_test, y_train, y_test, preprocessor, all_feature_names
    )
    metrics_list.append(gb_metrics)
    
    # Save the model
    joblib.dump(gb_model, os.path.join(output_dir, 'gradient_boosting_model.pkl'))
    
    # Compare models
    plot_model_comparison(metrics_list)
    
    print("\n=== Model Training and Evaluation Completed ===")
    print(f"Models and visualizations have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
