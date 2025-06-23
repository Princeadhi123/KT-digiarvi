import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create output directories
output_dir = 'eda_outputs'
regression_dir = os.path.join(output_dir, 'regression_analysis')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(regression_dir, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_data(filepath):
    """Load the preprocessed data."""
    return pd.read_csv(filepath)

def basic_info(df):
    """Display basic information about the dataset."""
    print("\n=== Dataset Information ===")
    print(f"Number of records: {len(df)}")
    print(f"Number of unique students: {df['student_id'].nunique()}")
    print(f"Number of unique exercises: {df['exercise_id'].nunique()}")
    print(f"Number of categories: {df['category'].nunique()}")
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

def analyze_categorical(df, column):
    """Analyze categorical columns."""
    print(f"\n=== {column} Distribution ===")
    value_counts = df[column].value_counts()
    print(value_counts)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y=column, order=value_counts.index)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'))
    plt.close()

def analyze_numerical(df, column):
    """Analyze numerical columns."""
    print(f"\n=== {column} Statistics ===")
    print(df[column].describe())
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=column, bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_analysis.png'))
    plt.close()

def analyze_student_performance(df):
    """Analyze student performance metrics."""
    # Performance by category
    plt.figure(figsize=(12, 6))
    category_performance = df.groupby('category')['score'].mean().sort_values(ascending=False)
    sns.barplot(x=category_performance.values, y=category_performance.index)
    plt.title('Average Score by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_performance.png'))
    plt.close()
    
    # Performance by grade
    plt.figure(figsize=(10, 6))
    grade_performance = df.groupby('grade')['score'].mean()
    sns.barplot(x=grade_performance.index, y=grade_performance.values)
    plt.title('Average Score by Grade')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grade_performance.png'))
    plt.close()

def analyze_learning_patterns(df):
    """Analyze learning patterns and progress over time."""
    # Calculate rolling average of scores
    df_sorted = df.sort_values(['student_id', 'order'])
    df_sorted['rolling_avg'] = df_sorted.groupby('student_id')['score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Plot learning curves for a sample of students
    sample_students = df_sorted['student_id'].drop_duplicates().sample(min(5, len(df_sorted['student_id'].unique())))
    plt.figure(figsize=(12, 6))
    
    for student in sample_students:
        student_data = df_sorted[df_sorted['student_id'] == student]
        plt.plot(student_data['order'], student_data['rolling_avg'], marker='o', label=f'Student {student}')
    
    plt.xlabel('Exercise Order')
    plt.ylabel('Rolling Average Score (window=5)')
    plt.title('Learning Progress for Sample Students')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_progress.png'))
    plt.close()

def analyze_time_spent(df):
    """Analyze time spent on exercises."""
    # Time spent vs score
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='time_spent', y='score', alpha=0.5)
    plt.title('Time Spent vs Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_score.png'))
    plt.close()
    
    # Time spent by category
    plt.figure(figsize=(12, 6))
    time_by_category = df.groupby('category')['time_spent'].median().sort_values(ascending=False)
    sns.barplot(x=time_by_category.values, y=time_by_category.index)
    plt.title('Median Time Spent by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_by_category.png'))
    plt.close()

def analyze_attempts(df):
    """Analyze attempt patterns."""
    # Pass rate by number of attempts
    plt.figure(figsize=(12, 6))
    attempt_success = df.groupby('total_attempts')['pass_status'].apply(
        lambda x: (x == 'Pass').mean()
    ).reset_index()
    
    sns.lineplot(data=attempt_success, x='total_attempts', y='pass_status', marker='o')
    plt.title('Pass Rate by Number of Attempts')
    plt.xlabel('Number of Attempts')
    plt.ylabel('Pass Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attempts_vs_pass_rate.png'))
    plt.close()

def analyze_correlations(df):
    """Analyze correlations between numerical variables."""
    print("\n=== Correlation Analysis ===")
    
    # Select numerical columns for correlation analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['student_id', 'order']]
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(regression_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Display top correlations
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs != 1]  # Remove self-correlations
    print("\nTop 10 Positive Correlations:")
    print(corr_pairs.head(10))
    
    print("\nTop 10 Negative Correlations:")
    print(corr_pairs.tail(10)[::-1])

def perform_regression_analysis(df):
    """Perform multiple linear regression analysis."""
    print("\n=== Regression Analysis ===")
    
    # Make a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values before encoding
    df_clean['sex'].fillna('Unknown', inplace=True)  # Fill missing sex with 'Unknown'
    df_clean['mean_perception'].fillna(df_clean['mean_perception'].mean(), inplace=True)
    
    # Create dummy variables for categorical columns
    df_encoded = pd.get_dummies(df_clean, columns=['category', 'sex'], drop_first=True)
    
    # Select features and target
    X = df_encoded.drop(['student_id', 'exercise_id', 'pass_status', 'score', 'pass_rate'], 
                       axis=1, errors='ignore')
    y = df_encoded['score']
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=['int64', 'float64', 'uint8'])
    
    # Standardize numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:  # Only standardize if there are numerical columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to numpy arrays to avoid pandas dtype issues
    X_train_np = X_train.astype(float).values
    y_train_np = y_train.astype(float).values
    
    # Add constant for statsmodels
    X_train_sm = sm.add_constant(X_train_np)
    
    # Fit OLS model
    model = sm.OLS(y_train_np, X_train_sm).fit()
    
    # Print model summary
    print("\n=== Linear Regression Results ===")
    print(model.summary())
    
    # Calculate VIF for multicollinearity
    vif_data = pd.DataFrame()
    feature_names = ['const'] + X.columns.tolist()
    vif_data["feature"] = feature_names
    
    # Calculate VIF for each feature
    vif_values = []
    for i in range(len(feature_names)):
        if i < X_train_sm.shape[1]:  # Ensure we don't go out of bounds
            vif = variance_inflation_factor(X_train_sm, i)
            vif_values.append(vif)
        else:
            vif_values.append(float('inf'))
    
    vif_data["VIF"] = vif_values
    
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data.sort_values('VIF', ascending=False).head(10))
    
    # Prepare test data for prediction
    X_test_np = X_test.astype(float).values
    X_test_sm = sm.add_constant(X_test_np)
    
    # Make predictions
    y_pred = model.predict(X_test_sm)
    
    # Convert y_test to numpy array for consistency
    y_test_np = y_test.astype(float).values
    
    # Plot regression results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, y_pred, alpha=0.5)
    plt.plot([y_test_np.min(), y_test_np.max()], 
             [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs Predicted Scores')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(regression_dir, 'regression_results.png'))
    plt.close()
    
    # Calculate and print metrics
    mse = mean_squared_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)
    
    print(f"\nModel Performance on Test Set:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_names = ['const'] + X.columns.tolist()
    coef = pd.DataFrame({
        'feature': feature_names[1:],  # Exclude constant
        'coefficient': model.params[1:],
        'p_value': model.pvalues[1:]
    })
    
    # Plot top 10 most important features
    plt.figure(figsize=(12, 8))
    coef = coef.sort_values('coefficient', key=abs, ascending=False).head(10)
    sns.barplot(data=coef, x='coefficient', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.axvline(0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(regression_dir, 'feature_importance.png'))
    plt.close()

def main():
    # Load data
    filepath = 'preprocessed_kt_data.csv'
    df = load_data(filepath)
    
    # Basic information
    basic_info(df)
    
    # Analyze categorical variables
    for col in ['category', 'grade', 'sex']:
        if col in df.columns:
            analyze_categorical(df, col)
    
    # Analyze numerical variables
    for col in ['score', 'time_spent', 'total_attempts', 'cumulative_passes', 'pass_rate']:
        if col in df.columns:
            analyze_numerical(df, col)
    
    # Advanced analysis
    analyze_student_performance(df)
    analyze_learning_patterns(df)
    analyze_time_spent(df)
    analyze_attempts(df)
    
    # Correlation analysis
    analyze_correlations(df)
    
    # Regression analysis
    perform_regression_analysis(df)
    
    print("\n=== EDA and Analysis Completed ===")
    print(f"Visualizations and results have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
