import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_indeed(job_title, location, num_pages=5):
    base_url = "https://www.indeed.com/jobs"
    jobs_data = []
    
    for page in range(num_pages):
        params = {
            'q': job_title,
            'l': location,
            'start': page * 10
        }
        
        response = requests.get(base_url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract job listings
        job_listings = soup.find_all('div', class_='jobsearch-SerpJobCard')
        
        for job in job_listings:
            title = job.find('a', class_='jobtitle').text.strip()
            company = job.find('span', class_='company').text.strip()
            location = job.find('div', class_='recJobLoc').get('data-rc-loc', '')
            
            # Extract salary if available
            salary_elem = job.find('span', class_='salaryText')
            salary = salary_elem.text.strip() if salary_elem else None
            
            # Extract job description to parse for skills
            job_url = 'https://www.indeed.com' + job.find('a', class_='jobtitle').get('href')
            job_response = requests.get(job_url)
            job_soup = BeautifulSoup(job_response.text, 'html.parser')
            description = job_soup.find('div', id='jobDescriptionText').text.strip()
            
            jobs_data.append({
                'title': title,
                'company': company,
                'location': location,
                'salary': salary,
                'description': description,
                'source': 'Indeed'
            })
    
    return pd.DataFrame(jobs_data)
