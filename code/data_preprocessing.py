def preprocess_job_data(df):
    # Handle missing values
    df['salary'] = df['salary'].fillna('Not specified')
    
    # Extract salary range and convert to numeric
    def extract_salary(salary_text):
        if salary_text == 'Not specified':
            return None
        # Extract numeric values using regex
        import re
        numbers = re.findall(r'\d+[,\d]*', salary_text)
        if len(numbers) >= 2:
            min_salary = float(numbers[0].replace(',', ''))
            max_salary = float(numbers[1].replace(',', ''))
            return (min_salary + max_salary) / 2
        elif len(numbers) == 1:
            return float(numbers[0].replace(',', ''))
        return None
    
    df['numeric_salary'] = df['salary'].apply(extract_salary)
    
    # Standardize job titles
    def standardize_title(title):
        title = title.lower()
        if 'data scientist' in title:
            return 'Data Scientist'
        elif 'data engineer' in title:
            return 'Data Engineer'
        # Add more standardizations as needed
        
    df['standardized_title'] = df['title'].apply(standardize_title)
    
    # Extract skills from job descriptions
    skills_list = ['python', 'java', 'sql', 'machine learning', 'tensorflow', 
                  'pytorch', 'aws', 'azure', 'docker', 'kubernetes', 
                  'tableau', 'power bi', 'excel', 'r', 'c++', 'javascript',
                  'react', 'node.js', 'hadoop', 'spark', 'nosql', 'mongodb']
    
    for skill in skills_list:
        df[f'requires_{skill}'] = df['description'].str.lower().str.contains(skill).astype(int)
    
    return df
