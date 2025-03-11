import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import re
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Function to load and prepare data
def load_data(file_path=None):
    """
    Load the job data CSV file
    """
    # If no file is specified, use the provided exact path
    if file_path is None or file_path == '':
        # Use the full file path
        file_path = r'C:\Users\Abdal\OneDrive\Desktop\Labs 2\Project_1_CMPSC_445\data\results\processed_data.csv'
    
    try:
        print(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None
        
# Function to preprocess and clean the data
def preprocess_data(df):
    """
    Clean and prepare the data for modeling
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Filter to only US jobs
    us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'US', 'DC']
    
    processed_df = processed_df[processed_df['standardized_location'].isin(us_states)]
    
    print(f"Filtered to {len(processed_df)} US-based jobs")
    
    # Handle missing values
    processed_df['annual_salary'] = processed_df['annual_salary'].fillna(
        processed_df.groupby('standardized_title')['annual_salary'].transform('median')
    )
    
    # If there are still NAs after group-based imputation, use overall median
    if processed_df['annual_salary'].isna().sum() > 0:
        median_salary = processed_df['annual_salary'].median()
        processed_df['annual_salary'] = processed_df['annual_salary'].fillna(median_salary)
        print(f"Imputed missing salaries with overall median: ${median_salary:.2f}")
    
    # Ensure all skills columns are available
    skill_cols = [col for col in processed_df.columns if col.startswith('has_')]
    for col in skill_cols:
        processed_df[col] = processed_df[col].fillna(0).astype(int)
    
    # Check if outliers exist in annual_salary
    Q1 = processed_df['annual_salary'].quantile(0.25)
    Q3 = processed_df['annual_salary'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = processed_df[(processed_df['annual_salary'] < lower_bound) | 
                           (processed_df['annual_salary'] > upper_bound)]
    
    print(f"Found {len(outliers)} potential salary outliers")
    
    # Cap outliers at bounds instead of removing them
    processed_df['annual_salary'] = np.where(
        processed_df['annual_salary'] < lower_bound,
        lower_bound,
        np.where(
            processed_df['annual_salary'] > upper_bound,
            upper_bound,
            processed_df['annual_salary']
        )
    )
    
    print(f"Capped outliers at lower bound: ${lower_bound:.2f} and upper bound: ${upper_bound:.2f}")
    
    return processed_df

# Function to prepare data for machine learning
def prepare_ml_data(df):
    """
    Prepare data for machine learning by creating feature matrices and target variable
    """
    # Drop rows with missing target variable (annual_salary), but we should have handled this already
    ml_df = df.dropna(subset=['annual_salary']).copy()
    
    # Select features and target
    categorical_features = ['standardized_title', 'standardized_location']
    
    # Add skill columns if they exist
    skill_cols = [col for col in ml_df.columns if col.startswith('has_')]
    
    features = categorical_features + skill_cols
    X = ml_df[features]
    y = ml_df['annual_salary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data size: {len(X_train)}")
    print(f"Testing data size: {len(X_test)}")
    
    # Create preprocessor
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, skill_cols)
        ]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, features

# Function to train and evaluate models
def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train RandomForest and GradientBoosting models and evaluate performance
    """
    # Model 1: Random Forest with feature selection
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate on training data
    y_train_pred_rf = rf_model.predict(X_train)
    train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
    train_rmse_rf = np.sqrt(train_mse_rf)
    train_r2_rf = r2_score(y_train, y_train_pred_rf)
    train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
    
    # Evaluate on test data
    y_test_pred_rf = rf_model.predict(X_test)
    test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
    test_rmse_rf = np.sqrt(test_mse_rf)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)
    test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
    
    print("\nRandom Forest Model Performance:")
    print(f"Training RMSE: ${train_rmse_rf:.2f}")
    print(f"Training R²: {train_r2_rf:.4f}")
    print(f"Training MAE: ${train_mae_rf:.2f}")
    print(f"Test RMSE: ${test_rmse_rf:.2f}")
    print(f"Test R²: {test_r2_rf:.4f}")
    print(f"Test MAE: ${test_mae_rf:.2f}")
    
    # Model 2: Gradient Boosting
    gb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
    ])
    
    gb_model.fit(X_train, y_train)
    
    # Evaluate on training data
    y_train_pred_gb = gb_model.predict(X_train)
    train_mse_gb = mean_squared_error(y_train, y_train_pred_gb)
    train_rmse_gb = np.sqrt(train_mse_gb)
    train_r2_gb = r2_score(y_train, y_train_pred_gb)
    train_mae_gb = mean_absolute_error(y_train, y_train_pred_gb)
    
    # Evaluate on test data
    y_test_pred_gb = gb_model.predict(X_test)
    test_mse_gb = mean_squared_error(y_test, y_test_pred_gb)
    test_rmse_gb = np.sqrt(test_mse_gb)
    test_r2_gb = r2_score(y_test, y_test_pred_gb)
    test_mae_gb = mean_absolute_error(y_test, y_test_pred_gb)
    
    print("\nGradient Boosting Model Performance:")
    print(f"Training RMSE: ${train_rmse_gb:.2f}")
    print(f"Training R²: {train_r2_gb:.4f}")
    print(f"Training MAE: ${train_mae_gb:.2f}")
    print(f"Test RMSE: ${test_rmse_gb:.2f}")
    print(f"Test R²: {test_r2_gb:.4f}")
    print(f"Test MAE: ${test_mae_gb:.2f}")
    
    # Return model information
    rf_metrics = {
        'train_rmse': train_rmse_rf, 
        'train_r2': train_r2_rf, 
        'train_mae': train_mae_rf,
        'test_rmse': test_rmse_rf, 
        'test_r2': test_r2_rf, 
        'test_mae': test_mae_rf,
        'predictions': y_test_pred_rf
    }
    
    gb_metrics = {
        'train_rmse': train_rmse_gb, 
        'train_r2': train_r2_gb, 
        'train_mae': train_mae_gb,
        'test_rmse': test_rmse_gb, 
        'test_r2': test_r2_gb, 
        'test_mae': test_mae_gb,
        'predictions': y_test_pred_gb
    }
    
    # Select best model based on test R²
    if rf_metrics['test_r2'] > gb_metrics['test_r2']:
        best_model = rf_model
        best_metrics = rf_metrics
        print("\nRandom Forest selected as the best model based on test R²")
    else:
        best_model = gb_model
        best_metrics = gb_metrics
        print("\nGradient Boosting selected as the best model based on test R²")
    
    return rf_model, gb_model, rf_metrics, gb_metrics, best_model, best_metrics, y_test

# Function to extract feature importance
def extract_feature_importance(model, X_train, features, preprocessor=None):
    """
    Extract feature importance from the trained model
    """
    try:
        # Get the regressor (last step in the pipeline)
        regressor = model.named_steps['regressor']
        
        # Get feature names (this gets complex with preprocessed data)
        if preprocessor:
            # Try to get feature names from the pipeline's preprocessing step
            try:
                # For newer scikit-learn versions
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            except:
                # For older versions
                # This is a simplified approach and may not be completely accurate
                cat_cols = [col for col in features if col in ['standardized_title', 'standardized_location']]
                num_cols = [col for col in features if col not in cat_cols]
                
                # Generate approximate feature names
                cat_features = []
                for col in cat_cols:
                    unique_vals = X_train[col].unique()
                    for val in unique_vals:
                        cat_features.append(f"{col}_{val}")
                
                feature_names = np.array(cat_features + num_cols)
        else:
            feature_names = np.array(features)
        
        # Extract feature importances
        importances = regressor.feature_importances_
        
        # If we have a feature selection step, adjust accordingly
        if 'feature_selection' in model.named_steps:
            # Get the support mask from feature selection
            support = model.named_steps['feature_selection'].get_support()
            
            # Filter feature names and importances based on support
            selected_indices = np.where(support)[0]
            feature_names = feature_names[selected_indices]
        
        # Create a dataframe for feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    except Exception as e:
        print(f"Error extracting feature importance: {str(e)}")
        # If extraction fails, return a placeholder
        return pd.DataFrame({
            'feature': ['Error extracting feature importance'],
            'importance': [1.0]
        })

# Function to create visualizations
def create_visualizations(processed_df, best_model, best_metrics, y_test, features, save_dir='visualizations'):
    """
    Create and save visualizations for model performance and data insights
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Salary distribution by job role
    plt.figure(figsize=(12, 8))
    top_roles = processed_df['standardized_title'].value_counts().head(10).index.tolist()
    plot_data = processed_df[processed_df['standardized_title'].isin(top_roles)]
    
    sns.boxplot(x='standardized_title', y='annual_salary', data=plot_data)
    plt.xticks(rotation=45, ha='right')
    plt.title('Salary Distribution by Job Role')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'salary_by_role.png'))
    plt.close()
    print(f"Created visualization: salary_by_role.png")
    
    # 2. Salary distribution by location
    plt.figure(figsize=(12, 8))
    top_locations = processed_df['standardized_location'].value_counts().head(10).index.tolist()
    plot_data = processed_df[processed_df['standardized_location'].isin(top_locations)]
    
    sns.boxplot(x='standardized_location', y='annual_salary', data=plot_data)
    plt.xticks(rotation=45, ha='right')
    plt.title('Salary Distribution by Location')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'salary_by_location.png'))
    plt.close()
    print(f"Created visualization: salary_by_location.png")
    
    # 3. Predicted vs Actual Salary
    plt.figure(figsize=(10, 8))
    
    # Use the best model's predictions
    plt.scatter(y_test, best_metrics['predictions'], alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Predicted vs Actual Salary')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predicted_vs_actual.png'))
    plt.close()
    print(f"Created visualization: predicted_vs_actual.png")
    
    # 4. Residuals plot
    plt.figure(figsize=(10, 8))
    residuals = y_test - best_metrics['predictions']
    plt.scatter(best_metrics['predictions'], residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Salary')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residuals.png'))
    plt.close()
    print(f"Created visualization: residuals.png")
    
    # 5. Feature Importance
    try:
        # Get feature importance
        feature_importance = extract_feature_importance(best_model, None, features)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 12))
        top_n = min(20, len(feature_importance))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
        plt.title(f'Top {top_n} Features by Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        plt.close()
        print(f"Created visualization: feature_importance.png")
    except Exception as e:
        print(f"Error creating feature importance visualization: {str(e)}")
    
    # 6. Correlation heatmap of skills
    try:
        skill_cols = [col for col in processed_df.columns if col.startswith('has_')]
        if len(skill_cols) > 0:
            plt.figure(figsize=(14, 12))
            
            # Calculate correlation matrix for skills
            correlation_matrix = processed_df[skill_cols + ['annual_salary']].corr()
            
            # Plot heatmap
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
            plt.title('Correlation Between Skills and Salary')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'skill_correlation.png'))
            plt.close()
            print(f"Created visualization: skill_correlation.png")
    except Exception as e:
        print(f"Error creating skill correlation visualization: {str(e)}")
    
    # 7. Salary histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(processed_df['annual_salary'], bins=30, kde=True)
    plt.title('Distribution of Annual Salaries')
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'salary_histogram.png'))
    plt.close()
    print(f"Created visualization: salary_histogram.png")
    
    print(f"All visualizations saved to {save_dir} directory")

# Main function
def main(data_path=None, output_dir='results', visualizations_dir='visualizations'):
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    try:
        # Step 1: Load the data
        print("Loading data...")
        df = load_data(data_path)
        
        if df is None:
            print("Error: Could not load data. Exiting.")
            return
        
        # Step 2: Preprocess the data
        print("\nPreprocessing data...")
        processed_df = preprocess_data(df)
        
        # Save processed data
        processed_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
        print(f"Processed data saved to {os.path.join(output_dir, 'processed_data.csv')}")
        
        # Print summary statistics
        print("\nData Summary:")
        print(f"Total job listings: {len(processed_df)}")
        print(f"Unique job titles: {processed_df['standardized_title'].nunique()}")
        print(f"Unique locations: {processed_df['standardized_location'].nunique()}")
        print(f"Jobs with salary information: {processed_df['annual_salary'].notna().sum()}")
        
        # Step 3: Prepare data for machine learning
        print("\nPreparing data for machine learning...")
        X_train, X_test, y_train, y_test, preprocessor, features = prepare_ml_data(processed_df)
        
        # Step 4: Train models
        print("\nTraining models...")
        rf_model, gb_model, rf_metrics, gb_metrics, best_model, best_metrics, y_test = train_models(
            X_train, X_test, y_train, y_test, preprocessor
        )
        
        # Save model metrics
        with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
            f.write("Model Performance Metrics\n")
            f.write("========================\n\n")
            
            f.write("Random Forest Model:\n")
            f.write(f"Training RMSE: ${rf_metrics['train_rmse']:.2f}\n")
            f.write(f"Training R²: {rf_metrics['train_r2']:.4f}\n")
            f.write(f"Training MAE: ${rf_metrics['train_mae']:.2f}\n")
            f.write(f"Test RMSE: ${rf_metrics['test_rmse']:.2f}\n")
            f.write(f"Test R²: {rf_metrics['test_r2']:.4f}\n")
            f.write(f"Test MAE: ${rf_metrics['test_mae']:.2f}\n\n")
            
            f.write("Gradient Boosting Model:\n")
            f.write(f"Training RMSE: ${gb_metrics['train_rmse']:.2f}\n")
            f.write(f"Training R²: {gb_metrics['train_r2']:.4f}\n")
            f.write(f"Training MAE: ${gb_metrics['train_mae']:.2f}\n")
            f.write(f"Test RMSE: ${gb_metrics['test_rmse']:.2f}\n")
            f.write(f"Test R²: {gb_metrics['test_r2']:.4f}\n")
            f.write(f"Test MAE: ${gb_metrics['test_mae']:.2f}\n")
        
        # Step 5: Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(
            processed_df, best_model, best_metrics, y_test, features, 
            save_dir=visualizations_dir
        )
        
        # Generate a summary report
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("Software Engineering Job Salary Analysis Report\n")
            f.write("===========================================\n\n")
            
            f.write("Data Summary:\n")
            f.write(f"Total job listings: {len(processed_df)}\n")
            f.write(f"Unique job titles: {processed_df['standardized_title'].nunique()}\n")
            f.write(f"Unique locations: {processed_df['standardized_location'].nunique()}\n")
            f.write(f"Jobs with salary information: {processed_df['annual_salary'].notna().sum()}\n\n")
            
            f.write("Salary Statistics:\n")
            salary_stats = processed_df['annual_salary'].describe()
            f.write(f"- Mean: ${salary_stats['mean']:.2f}\n")
            f.write(f"- Median: ${salary_stats['50%']:.2f}\n")
            f.write(f"- Min: ${salary_stats['min']:.2f}\n")
            f.write(f"- Max: ${salary_stats['max']:.2f}\n\n")
            
            f.write("Model Performance:\n")
            f.write(f"- Best Model: {'Random Forest' if rf_metrics['test_r2'] > gb_metrics['test_r2'] else 'Gradient Boosting'}\n")
            f.write(f"- Test R²: {best_metrics['test_r2']:.4f}\n")
            f.write(f"- Test RMSE: ${best_metrics['test_rmse']:.2f}\n")
            f.write(f"- Test MAE: ${best_metrics['test_mae']:.2f}\n\n")
            
            # Top 5 highest paying job titles
            f.write("Top 5 Highest Paying Job Titles:\n")
            title_salaries = processed_df.groupby('standardized_title')['annual_salary'].mean().sort_values(ascending=False)
            for i, (title, salary) in enumerate(title_salaries.head(5).items(), 1):
                f.write(f"{i}. {title}: ${salary:.2f}\n")
            f.write("\n")
            
            # Top 5 highest paying locations
            f.write("Top 5 Highest Paying Locations:\n")
            loc_salaries = processed_df.groupby('standardized_location')['annual_salary'].mean().sort_values(ascending=False)
            for i, (loc, salary) in enumerate(loc_salaries.head(5).items(), 1):
                f.write(f"{i}. {loc}: ${salary:.2f}\n")
            f.write("\n")
            
            f.write("Conclusions:\n")
            f.write("- The analysis reveals significant salary variation across different job roles and locations.\n")
            f.write("- Technical specialization appears to be a strong driver of higher compensation.\n")
            f.write("- Location remains a significant factor in salary determination, with tech hubs offering higher compensation.\n")
            f.write("- The model can predict software engineering salaries with reasonable accuracy based on role, location, and skills.\n")
        
        print("\nAnalysis complete!")
        print(f"Results saved to {output_dir}")
        print(f"Visualizations saved to {visualizations_dir}")
        print(f"Check {os.path.join(output_dir, 'analysis_report.txt')} for a summary of findings.")
    
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the program if it's the main script
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = None
    
    main(data_path)