from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

def scrape_glassdoor_jobs():
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    url = "https://www.glassdoor.com/Job/los-angeles-ca-us-software-engineer-jobs-SRCH_IL.0,17_IC1146821_KO18,35.htm"
    driver.get(url)
    
    time.sleep(5)  # Allow time for the page to load
    
    jobs = driver.find_elements(By.CLASS_NAME, "jobCard")[:3]  # Get first 3 job cards
    job_list = []
    
    for job in jobs:
        try:
            title = job.find_element(By.CLASS_NAME, "jobTitle").text
            company = job.find_element(By.CLASS_NAME, "companyName").text
            location = job.find_element(By.CLASS_NAME, "jobLocation").text
            try:
                salary = job.find_element(By.CLASS_NAME, "salaryEstimate").text
            except:
                salary = "Not Provided"
            
            job_list.append({
                "title": title,
                "company": company,
                "location": location,
                "salary": salary
            })
        except Exception as e:
            print("Error extracting job details:", e)
    
    driver.quit()
    
    return job_list

# Run the scraper and print results
jobs = scrape_glassdoor_jobs()
for job in jobs:
    print(job)
