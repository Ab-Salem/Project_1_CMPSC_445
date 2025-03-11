# Job Market Analysis for Salary Prediction and Skill Identification

This project analyzes job market data to predict salaries and identify important skills in the fields of computer science, data science, and artificial intelligence.

## Project Description

This data science project focuses on analyzing job posting data from various sources to:

1. Predict salaries for different job roles in tech fields using machine learning
2. Identify the most important skills for different job roles
3. Visualize salary distributions and skill importance across different locations and job roles

The analysis uses scraped job data from websites like Indeed, LinkedIn, Glassdoor, USA Jobs, and FlexJobs, focusing on roles in computer science, data science, and AI.

## Project Setup

### Installation Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements.txt file should include:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
plotly
wordcloud
```

### Directory Structure

The project expects the following directory structure:
```
Project_1_CMPSC_445/
├── data/                      # Directory containing CSV files with job data
│   ├── process_predict_visualize.py  # Main analysis script
│   ├── *.csv 
    ├── results/                   # Output directory for results (created automatically)
    ├── visualizations/                 # job data CSV files  
├── FlexJobs_scraper/                      
│   ├── main.py  # Main scraper
│   ├── *.csv 
├── USAJOBS_scraper/                     
│   ├── main.py  # Main analysis script
│   ├── *.csv          
```

## Running the Analysis

### Important: Running the Script

Navigate to the `data` directory where the script is located:

```bash
cd data
```

Then run the script with:

```bash
python process_predict_visualize.py --data-dir .
```

This tells the script to look for CSV files in the current directory.

### Command Line Arguments

The script accepts the following arguments:
```
--data-dir: Directory containing CSV files (default: current directory when not specified)
--output-dir: Directory to save results (default: 'results')
--viz-dir: Directory to save visualizations (default: 'visualizations')
```

Example with all arguments specified:
```bash
python process_predict_visualize.py --data-dir . --output-dir ../results --viz-dir ../visualizations
```

## Data Processing Pipeline

### 1. Data Collection
- The script loads and combines all CSV files in the specified directory
- Each file is expected to contain job posting data with fields like title, location, company/agency, salary, etc.
- Missing data is handled through robust preprocessing

### 2. Data Preprocessing
- Standardizes job titles into consistent categories
- Cleans and standardizes location data
- Extracts and normalizes salary information
- Identifies skills mentioned in job descriptions
- Creates binary skill indicators for each job posting
- Generates synthetic data for missing values when necessary

### 3. Feature Engineering
- Converts categorical data (job titles, locations, companies) into numerical features
- Extracts salary value and period (hourly, annually, etc.)
- Converts all salaries to annual equivalents for comparison
- Creates skill indicator features

### 4. Model Development
- **Model 1: Random Forest Regressor**
  - Uses job title, location, company, and skills as features
  - Predicts annual salary
  
- **Model 2: Gradient Boosting Regressor**
  - Same features as Model 1
  - Provides an alternative prediction approach

### 5. Skill Importance Analysis
- Extracts feature importance from models to identify valuable skills
- Analyzes skill importance by job role
- Identifies the most in-demand skills for each job category

### 6. Visualization
- Generates charts and graphs to visualize findings
- Creates salary distribution visualizations
- Maps skill importance across job roles and locations

## Results and Findings

Our analysis of software engineering job market data revealed several key insights:

### Data Overview
- Analyzed 3,345 total job listings, with 1,543 US-based positions
- Identified 20 distinct standardized job titles across 45 different locations
- All records contained valid salary information after preprocessing

### Salary Prediction Models
- **Gradient Boosting Model** outperformed Random Forest with:
  - Test R² score of 0.2839 (vs. 0.2215 for Random Forest)
  - Test RMSE of $32,971.33
  - Test MAE of $22,771.63
- The predictive power (R² of ~0.28) indicates that while our model captures some salary variance, other factors not present in our dataset also significantly influence salaries

### Salary Distribution Findings
- Salary distribution shows considerable variation across job roles
- Software engineering roles in tech hubs (CA, WA, NY) commanded higher compensation
- Identified and capped 137 salary outliers to improve model performance
- Median annual salary across all roles: approximately $123,600

### Skills Analysis
- Created correlation heatmap showing relationships between skills and salary
- Most influential skills for salary determination were identified through feature importance analysis
- Technical specialization appears to be a strong driver of higher compensation
- Skills like cloud technologies, machine learning frameworks, and specialized programming languages correlated with higher salaries

### Regional Insights
- Location remains a significant factor in salary determination
- Tech hubs offered notably higher compensation packages
- Remote positions showed competitive salaries comparable to major metro areas

These findings provide valuable insights for job seekers, employers, and educational institutions in the software engineering field. The prediction model can be used to estimate fair market salaries based on role, location, and skill set.

## Outputs

The analysis generates the following outputs:

### Data Files
- `raw_data.csv`: Combined raw data from all input files
- `processed_data.csv`: Preprocessed data with standardized fields

### Results
- `model_metrics.txt`: Performance metrics for both salary prediction models
- `skill_importance.txt`: Importance scores for skills by job role
- `analysis_report.txt`: Comprehensive report with key findings and conclusions

### Visualizations
- Salary distributions by job role and location (box plots)
- Predicted vs. actual salary comparison
- Feature importance charts
- Skill importance for top job roles
- Heatmap of skills across roles
- Salary histogram
- Map of average salaries by location
- Word cloud of most common skills


## Future Improvements

- Collect additional data with more salary information
- Improve skill extraction with advanced NLP techniques
- Add time-based analysis to track changing skill demands
- Create interactive visualizations for better exploration
- Integrate with live data sources for continuous updates
- Enhance predictive models to achieve higher R² scores
- Expand analysis to include additional factors like experience level and education requirements

  ##Visuals
  ![image](https://github.com/user-attachments/assets/94941ae7-a3f0-4fbf-9771-ef785231ea49)
  ![image](https://github.com/user-attachments/assets/14646269-2877-41a5-a864-d889e973055e)
  ![image](https://github.com/user-attachments/assets/446b742d-e6da-4cb1-adae-606c3ae1b174)
  ![image](https://github.com/user-attachments/assets/df6d2310-05a6-42e3-814e-48e59a4989bd)
  ![image](https://github.com/user-attachments/assets/8e70ec35-4b17-4412-a887-e18f2d50fe90)
  ![image](https://github.com/user-attachments/assets/a1350381-cce9-4313-9f8d-72cb651feae0)





