

def engineer_features(df):
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['standardized_title'], prefix='title')
    
    # Extract location features
    df['is_remote'] = df['location'].str.lower().str.contains('remote').astype(int)
    
    # Group locations by state/region
    def extract_state(location):
        # Simple state extraction - would need to be more sophisticated in practice
        states = {
            'NY': 'New York',
            'CA': 'California',
            # Add more state mappings
        }
        for code, name in states.items():
            if code in location or name in location:
                return code
        return 'Other'
    
    df['state'] = df['location'].apply(extract_state)
    df = pd.get_dummies(df, columns=['state'], prefix='state')
    
    # Create experience level feature based on job title
    df['is_senior'] = df['title'].str.lower().str.contains('senior|sr\.?|lead|principal').astype(int)
    
    # Create company size feature if available
    # This would require additional data collection
    
    return df
