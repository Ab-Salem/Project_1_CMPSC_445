#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Job Market Analysis Project for Salary Prediction and Skill Identification.
This script orchestrates the entire workflow including data collection, preprocessing,
feature engineering, model training, evaluation, and visualization.
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import custom modules for each stage of the pipeline
from data_collection import (
    scrape_indeed, 
    scrape_linkedin, 
    scrape_glassdoor,
    scrape_simplyhired
)
from data_preprocessing import (
    clean_job_data,
    standardize_job_titles,
    extract_salary_info,
    extract_skills
)
from feature_engineering import (
    engineer_features,
    create_location_features,
    create_company_features,
    create_skill_features
)
from visualizations import (
    plot_salary_distribution,
    plot_salary_by_location,
    plot_salary_by_job_role,
    plot_skill_importance,
    plot_skill_heatmap,
    create_interactive_map
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("job_market_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
def create_directories():
    """Create necessary directories for the project"""
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'visualizations',
        'reports'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def collect_data(job_titles, locations, num_samples=100):
    """
    Collect job data from multiple sources
    
    Args:
        job_titles (list): List of job titles to search for
        locations (list): List of locations to search in
        num_samples (int): Number of samples to collect per job title and location
        
    Returns:
        pd.DataFrame: Combined dataframe of all collected job data
    """
    logger.info("Starting data collection...")
    all_jobs = []
    
    for job_title in tqdm(job_titles, desc="Processing job titles"):
        for location in tqdm(locations, desc="Processing locations", leave=False):
            try:
                # Collect data from Indeed
                logger.info(f"Collecting data from Indeed for {job_title} in {location}")
                indeed_jobs = scrape_indeed(job_title, location, num_pages=num_samples//10)
                all_jobs.append(indeed_jobs)
                
                # Collect data from LinkedIn
                logger.info(f"Collecting data from LinkedIn for {job_title} in {location}")
                linkedin_jobs = scrape_linkedin(job_title, location, num_pages=num_samples//10)
                all_jobs.append(linkedin_jobs)
                
                # Collect data from Glassdoor
                logger.info(f"Collecting data from Glassdoor for {job_title} in {location}")
                glassdoor_jobs = scrape_glassdoor(job_title, location, num_pages=num_samples//10)
                all_jobs.append(glassdoor_jobs)
                
                # Collect data from SimplyHired
                logger.info(f"Collecting data from SimplyHired for {job_title} in {location}")
                simplyhired_jobs = scrape_simplyhired(job_title, location, num_pages=num_samples//10)
                all_jobs.append(simplyhired_jobs)
                
                # Avoid being blocked by adding delay
                time.sleep(5)  
            except Exception as e:
                logger.error(f"Error collecting data for {job_title} in {location}: {e}")
    
    # Combine all job dataframes
    combined_df = pd.concat(all_jobs, ignore_index=True)
    logger.info(f"Data collection completed. Total jobs collected: {len(combined_df)}")
    
    # Save raw data
    raw_data_path = 'data/raw/job_data_raw.csv'
    combined_df.to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    return combined_df

def preprocess_data(df):
    """
    Preprocess the collected job data
    
    Args:
        df (pd.DataFrame): Raw job data
        
    Returns:
        pd.DataFrame: Preprocessed job data
    """
    logger.info("Starting data preprocessing...")
    
    # Clean job data
    df = clean_job_data(df)
    logger.info("Basic data cleaning completed")
    
    # Standardize job titles
    df = standardize_job_titles(df)
    logger.info("Job titles standardized")
    
    # Extract salary information
    df = extract_salary_info(df)
    logger.info("Salary information extracted")
    
    # Extract skills from job descriptions
    df = extract_skills(df)
    logger.info("Skills extracted from job descriptions")
    
    # Save preprocessed data
    processed_data_path = 'data/processed/job_data_processed.csv'
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Preprocessed data saved to {processed_data_path}")
    
    return df

def create_features(df):
    """
    Create features for model training
    
    Args:
        df (pd.DataFrame): Preprocessed job data
        
    Returns:
        pd.DataFrame: Feature-engineered job data
    """
    logger.info("Starting feature engineering...")
    
    # Create basic features
    df = engineer_features(df)
    logger.info("Basic features created")
    
    # Create location-based features
    df = create_location_features(df)
    logger.info("Location features created")
    
    # Create company-based features
    df = create_company_features(df)
    logger.info("Company features created")
    
    # Create skill-based features
    df = create_skill_features(df)
    logger.info("Skill features created")
    
    # Save feature-engineered data
    feature_data_path = 'data/processed/job_data_featured.csv'
    df.to_csv(feature_data_path, index=False)
    logger.info(f"Feature-engineered data saved to {feature_data_path}")
    
    return df

def train_models(df, target='salary_normalized', test_size=0.2, random_state=42):
    """
    Train and evaluate salary prediction models
    
    Args:
        df (pd.DataFrame): Feature-engineered job data
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (trained_models, model_metrics, feature_names)
    """
    logger.info("Starting model training and evaluation...")
    
    # Prepare features and target
    # Drop rows with missing target values
    df_model = df.dropna(subset=[target])
    
    # Select features
    feature_cols = [col for col in df_model.columns if col.startswith(('skill_', 'title_', 'location_', 'company_'))]
    X = df_model[feature_cols]
    y = df_model[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples)")
    
    # Define models
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }
    
    # Train and evaluate models
    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        model_metrics[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        logger.info(f"{name} evaluation metrics: Test MAE={test_mae:.2f}, Test RMSE={test_rmse:.2f}, Test R²={test_r2:.2f}")
    
    # Save models
    for name, model in trained_models.items():
        model_path = f'models/{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model {name} saved to {model_path}")
    
    # Save model metrics
    metrics_df = pd.DataFrame.from_dict({(i, j): model_metrics[i][j] 
                                       for i in model_metrics.keys() 
                                       for j in model_metrics[i].keys()},
                                      orient='index')
    metrics_df.to_csv('models/model_metrics.csv')
    logger.info("Model metrics saved to models/model_metrics.csv")
    
    return trained_models, model_metrics, X.columns

def analyze_skill_importance(models, feature_names):
    """
    Analyze skill importance from trained models
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (pd.Index): Names of features used in the models
        
    Returns:
        dict: Dictionary of skill importance dataframes for each model
    """
    logger.info("Analyzing skill importance...")
    
    skill_importance_dict = {}
    
    for name, model in models.items():
        # Skip models that don't support feature importance
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            continue
            
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
        else:
            # For linear models
            importances = np.abs(model.coef_)
            
        # Create DataFrame of feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Filter for skill features only
        skill_importance = importance_df[importance_df['Feature'].str.startswith('skill_')].copy()
        skill_importance['Skill'] = skill_importance['Feature'].str.replace('skill_', '')
        skill_importance = skill_importance.sort_values('Importance', ascending=False)
        
        skill_importance_dict[name] = skill_importance
        
        # Save skill importance
        skill_importance.to_csv(f'models/skill_importance_{name}.csv', index=False)
        logger.info(f"Skill importance for {name} saved to models/skill_importance_{name}.csv")
    
    return skill_importance_dict

def create_visualizations(df, models, model_metrics, skill_importance_dict):
    """
    Create visualizations for the project
    
    Args:
        df (pd.DataFrame): Feature-engineered job data
        models (dict): Dictionary of trained models
        model_metrics (dict): Dictionary of model metrics
        skill_importance_dict (dict): Dictionary of skill importance dataframes
    """
    logger.info("Creating visualizations...")
    
    # 1. Salary distribution
    plot_salary_distribution(df, save_path='visualizations/salary_distribution.png')
    logger.info("Salary distribution plot created")
    
    # 2. Salary by location
    plot_salary_by_location(df, save_path='visualizations/salary_by_location.png')
    logger.info("Salary by location plot created")
    
    # 3. Salary by job role
    plot_salary_by_job_role(df, save_path='visualizations/salary_by_job_role.png')
    logger.info("Salary by job role plot created")
    
    # 4. Interactive salary map
    create_interactive_map(df, save_path='visualizations/salary_map.html')
    logger.info("Interactive salary map created")
    
    # 5. Skill importance for each model
    for name, skill_importance in skill_importance_dict.items():
        plot_skill_importance(skill_importance, model_name=name, 
                              save_path=f'visualizations/skill_importance_{name}.png')
    logger.info("Skill importance plots created")
    
    # 6. Skill importance heatmap across job roles
    plot_skill_heatmap(df, skill_importance_dict, 
                       save_path='visualizations/skill_heatmap.png')
    logger.info("Skill importance heatmap created")
    
    # 7. Model comparison
    plt.figure(figsize=(10, 6))
    metrics = pd.DataFrame.from_dict({(i, j): model_metrics[i][j] 
                                    for i in model_metrics.keys() 
                                    for j in model_metrics[i].keys()},
                                   orient='index')
    
    # Plot test metrics
    test_metrics = pd.DataFrame({
        'Model': list(model_metrics.keys()),
        'MAE': [model_metrics[model]['test_mae'] for model in model_metrics],
        'RMSE': [model_metrics[model]['test_rmse'] for model in model_metrics],
        'R²': [model_metrics[model]['test_r2'] for model in model_metrics]
    })
    
    test_metrics_melted = pd.melt(test_metrics, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    sns.barplot(x='Model', y='Value', hue='Metric', data=test_metrics_melted)
    plt.title('Model Performance Comparison (Test Set)')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    logger.info("Model comparison plot created")

def generate_report():
    """Generate a simple summary report"""
    logger.info("Generating summary report...")
    
    report_path = 'reports/summary_report.md'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load model metrics
    try:
        model_metrics = pd.read_csv('models/model_metrics.csv', index_col=0)
    except:
        model_metrics = pd.DataFrame({'Model': [], 'Metric': [], 'Value': []})
    
    with open(report_path, 'w') as f:
        f.write(f"# Job Market Analysis Project Summary\n\n")
        f.write(f"Report generated on: {timestamp}\n\n")
        
        f.write("## Data Collection Summary\n\n")
        try:
            raw_data = pd.read_csv('data/raw/job_data_raw.csv')
            f.write(f"- Total job postings collected: {len(raw_data)}\n")
            f.write(f"- Data sources: {', '.join(raw_data['source'].unique())}\n")
            f.write(f"- Job roles: {', '.join(raw_data['job_title'].unique())}\n")
            f.write(f"- Locations: {', '.join(raw_data['location'].unique())}\n\n")
        except:
            f.write("- Raw data not found or could not be loaded\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Test MAE | Test RMSE | Test R² |\n")
        f.write("|-------|---------|-----------|--------|\n")
        
        for model in model_metrics.index.levels[0] if isinstance(model_metrics.index, pd.MultiIndex) else []:
            mae = model_metrics.loc[(model, 'test_mae')][0] if isinstance(model_metrics.index, pd.MultiIndex) else 'N/A'
            rmse = model_metrics.loc[(model, 'test_rmse')][0] if isinstance(model_metrics.index, pd.MultiIndex) else 'N/A'
            r2 = model_metrics.loc[(model, 'test_r2')][0] if isinstance(model_metrics.index, pd.MultiIndex) else 'N/A'
            f.write(f"| {model} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |\n")
        
        f.write("\n## Top Skills by Importance\n\n")
        
        try:
            # Use Random Forest skill importance if available
            skill_importance = pd.read_csv('models/skill_importance_random_forest.csv')
            top_skills = skill_importance.head(10)
            
            f.write("| Skill | Importance |\n")
            f.write("|-------|------------|\n")
            for _, row in top_skills.iterrows():
                f.write(f"| {row['Skill']} | {row['Importance']:.4f} |\n")
        except:
            f.write("- Skill importance data not found or could not be loaded\n")
    
    logger.info(f"Summary report generated at {report_path}")

def main():
    """Main function to run the entire pipeline"""
    start_time = time.time()
    logger.info("Starting Job Market Analysis pipeline")
    
    # Create necessary directories
    create_directories()
    
    # Define job titles and locations to search for
    job_titles = [
        "Data Scientist", 
        "Machine Learning Engineer",
        "Data Engineer",
        "Software Engineer",
        "Frontend Developer",
        "Backend Developer",
        "DevOps Engineer",
        "AI Engineer",
        "Full Stack Developer",
        "Data Analyst"
    ]
    
    locations = [
        "New York, NY",
        "San Francisco, CA",
        "Seattle, WA",
        "Austin, TX",
        "Boston, MA",
        "Chicago, IL",
        "Los Angeles, CA",
        "Denver, CO",
        "Remote"
    ]
    
    # Step 1: Data Collection
    # Uncomment the line below to run data collection (it takes time)
    # raw_df = collect_data(job_titles, locations, num_samples=50)
    
    # For development/testing, load existing data if available
    try:
        raw_df = pd.read_csv('data/raw/job_data_raw.csv')
        logger.info(f"Loaded existing raw data with {len(raw_df)} records")
    except FileNotFoundError:
        logger.warning("No existing raw data found. Please uncomment the data collection step.")
        raw_df = collect_data(job_titles[:2], locations[:2], num_samples=10)  # Collect minimal data for testing
    
    # Step 2: Data Preprocessing
    processed_df = preprocess_data(raw_df)
    
    # Step 3: Feature Engineering
    featured_df = create_features(processed_df)
    
    # Step 4: Model Training and Evaluation
    models, metrics, feature_names = train_models(featured_df)
    
    # Step 5: Skill Importance Analysis
    skill_importance_dict = analyze_skill_importance(models, feature_names)
    
    # Step 6: Create Visualizations
    create_visualizations(featured_df, models, metrics, skill_importance_dict)
    
    # Step 7: Generate Summary Report
    generate_report()
    
    end_time = time.time()
    logger.info(f"Job Market Analysis pipeline completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
