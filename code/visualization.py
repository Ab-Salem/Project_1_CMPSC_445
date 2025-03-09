def create_visualizations(df, models, skill_importance):
    # 1. Salary distribution by job role
    plt.figure(figsize=(12, 6))
    job_roles = [col.replace('title_', '') for col in df.columns if col.startswith('title_')]
    
    for role in job_roles:
        role_data = df[df[f'title_{role}'] == 1]['numeric_salary'].dropna()
        if len(role_data) > 0:
            sns.histplot(role_data, label=role, alpha=0.5, kde=True)
    
    plt.title('Salary Distribution by Job Role')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('salary_distribution.png')
    
    # 2. Salary by location
    plt.figure(figsize=(12, 6))
    locations = df['state'].unique()
    
    salary_by_location = []
    for loc in locations:
        loc_data = df[df['state'] == loc]['numeric_salary'].dropna()
        if len(loc_data) > 0:
            salary_by_location.append({
                'Location': loc,
                'Average Salary': loc_data.mean(),
                'Count': len(loc_data)
            })
    
    loc_df = pd.DataFrame(salary_by_location)
    sns.barplot(x='Location', y='Average Salary', data=loc_df)
    plt.title('Average Salary by Location')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('salary_by_location.png')
    
    # 3. Skill importance heatmap
    plt.figure(figsize=(12, 10))
    # This would need to be extended to show importance by job role
    
    sns.heatmap(skill_importance.pivot_table(index='Skill', columns='Job Role', values='Importance'),
               cmap='viridis', annot=True)
    plt.title('Skill Importance by Job Role')
    plt.tight_layout()
    plt.savefig('skill_importance.png')
