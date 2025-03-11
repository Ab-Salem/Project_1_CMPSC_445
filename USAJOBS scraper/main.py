import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import os
from datetime import datetime

def setup_driver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    # Uncomment the next line to run in headless mode
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def search_jobs(driver, keyword, location=""):
    """Navigate to USAJOBS.gov and search for specified jobs."""
    print(f"Searching for '{keyword}' jobs...")
    
    try:
        # Navigate to USAJOBS.gov
        driver.get("https://www.usajobs.gov/")
        
        # Wait for the search box to be available
        keyword_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "nav-keyword"))
        )
        
        # Enter search keyword
        keyword_input.clear()
        keyword_input.send_keys(keyword)
        
        # Enter location if provided
        if location:
            location_input = driver.find_element(By.ID, "nav-location")
            location_input.clear()
            location_input.send_keys(location)
        
        # Submit the search form by pressing Enter
        keyword_input.send_keys(Keys.RETURN)
        
        # Wait for search results to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "usajobs-search-results"))
        )
        
        # Wait a bit more to ensure all results are loaded
        time.sleep(3)
        
        print(f"Search completed successfully for '{keyword}'!")
        return True
        
    except Exception as e:
        print(f"Error during search for '{keyword}': {str(e)}")
        print(f"Current URL: {driver.current_url}")
        return False

def extract_job_details(driver, max_jobs=100):
    """Extract details from job listings on the current page."""
    print(f"Extracting details for up to {max_jobs} jobs...")
    
    jobs_data = []
    
    try:
        # Wait for the search results to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "usajobs-search-results"))
        )
        
        # Pause to let all results fully load
        time.sleep(2)
        
        # Get job cards
        job_cards = driver.find_elements(By.CSS_SELECTOR, ".usajobs-search-result--core")
        print(f"Found {len(job_cards)} job cards on the current page")
        
        # Process each job card
        for i, card in enumerate(job_cards[:max_jobs]):
            try:
                job_data = {}
                
                # Extract job title
                title_element = card.find_element(By.CSS_SELECTOR, "h2 a.usajobs-search-result--core__title")
                job_data["title"] = title_element.text.strip()
                job_data["url"] = title_element.get_attribute("href")
                
                # Extract agency
                try:
                    agency_element = card.find_element(By.CSS_SELECTOR, ".usajobs-search-result--core__agency")
                    job_data["agency"] = agency_element.text.strip()
                except NoSuchElementException:
                    job_data["agency"] = "Not specified"
                
                # Extract location
                try:
                    location_element = card.find_element(By.CSS_SELECTOR, ".usajobs-search-result--core__location")
                    job_data["location"] = location_element.text.strip()
                except NoSuchElementException:
                    job_data["location"] = "Not specified"
                
                # Extract salary
                try:
                    salary_element = card.find_element(By.CSS_SELECTOR, ".usajobs-search-result--core__salary")
                    job_data["salary"] = salary_element.text.strip()
                except NoSuchElementException:
                    job_data["salary"] = "Not specified"
                
                # Extract closing date
                try:
                    closing_element = card.find_element(By.CSS_SELECTOR, ".usajobs-search-result--core__closing-date")
                    job_data["closing_date"] = closing_element.text.strip()
                except NoSuchElementException:
                    job_data["closing_date"] = "Not specified"
                
                # Add search keyword for reference
                job_data["search_term"] = driver.find_element(By.ID, "nav-keyword").get_attribute("value")
                
                # Add the job to our results
                jobs_data.append(job_data)
                print(f"Successfully extracted job {i+1}: {job_data['title']}")
                
            except StaleElementReferenceException:
                print(f"Encountered stale element for job {i+1}, skipping")
                continue
                
            except Exception as e:
                print(f"Error extracting job {i+1}: {str(e)}")
                continue
        
    except Exception as e:
        print(f"Error in extract_job_details: {str(e)}")
    
    return jobs_data

def check_for_next_page(driver):
    """Check if there's a next page of search results and navigate to it if available."""
    try:
        # Look for the next page button
        next_page_buttons = driver.find_elements(By.CSS_SELECTOR, ".usajobs-search-pagination__next-page-link")
        
        if next_page_buttons and len(next_page_buttons) > 0:
            # Check if the next button is disabled
            if "is-disabled" in next_page_buttons[0].get_attribute("class"):
                print("No more pages available")
                return False
            
            print("Navigating to next page...")
            next_page_buttons[0].click()
            
            # Wait for the next page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "usajobs-search-results"))
            )
            
            # Wait a bit more to ensure all results are loaded
            time.sleep(3)
            
            return True
        else:
            print("No next page button found")
            return False
    
    except Exception as e:
        print(f"Error checking for next page: {str(e)}")
        return False

def save_to_csv(jobs_data, filename="usajobs_tech_jobs.csv"):
    """Save job data to a CSV file."""
    if not jobs_data:
        print("No job data to save.")
        return
    
    # Create a timestamp string for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create csv_exports directory if it doesn't exist
    os.makedirs("csv_exports", exist_ok=True)
    
    # Create the full filepath
    filepath = os.path.join("csv_exports", f"{filename.split('.')[0]}_{timestamp}.csv")
    
    df = pd.DataFrame(jobs_data)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return filepath

def main():
    """Main function to run the scraper."""
    driver = setup_driver()
    all_jobs_data = []
    
    # Define the job categories and titles to search for
    job_categories = {
        "Software Engineering & Development": [
            "Software Engineer", "Full-Stack Developer", "Front-End Developer", 
            "Back-End Developer", "Web Developer", "Mobile App Developer", 
            "Embedded Systems Engineer", "Game Developer", "DevOps Engineer", 
            "Firmware Engineer"
        ],
        "Data Science & AI": [
            "Data Scientist", "Machine Learning Engineer", "AI Engineer", 
            "Deep Learning Engineer", "NLP Engineer", "Computer Vision Engineer", 
            "Data Engineer", "Big Data Engineer", "Business Intelligence Developer", 
            "Quantitative Analyst"
        ],
        "Cybersecurity & IT": [
            "Cybersecurity Analyst", "Ethical Hacker", "Penetration Tester", 
            "Security Engineer", "Network Security Engineer", "Information Security Analyst", 
            "IT Support Specialist", "System Administrator", "Network Administrator", 
            "Cloud Security Engineer", "Incident Response Analyst"
        ],
        "Cloud Computing & DevOps": [
            "Cloud Engineer", "AWS Solutions Architect", "Azure Engineer", 
            "Site Reliability Engineer", "SRE", "Kubernetes Engineer", 
            "Cloud Infrastructure Engineer", "Platform Engineer", 
            "IT Operations Engineer", "DevOps Manager", "Cloud Consultant"
        ],
        "Software Testing & Quality Assurance": [
            "QA Engineer", "Software Tester", "Test Automation Engineer", 
            "Performance Tester", "Quality Assurance Analyst"
        ],
        "Database & Backend Technologies": [
            "Database Administrator", "DBA", "SQL Developer", 
            "NoSQL Database Engineer", "Data Warehouse Engineer", "ETL Developer"
        ],
        "IT Project Management & Business Analysis": [
            "IT Project Manager", "Scrum Master", "Agile Coach", 
            "Business Analyst", "Product Manager"
        ],
        "Other IT & Engineering Roles": [
            "UI UX Designer", "Technical Writer", "Embedded Software Engineer", 
            "IT Consultant", "IT Auditor"
        ]
    }
    
    try:
        # Flatten the job titles list
        all_job_titles = []
        for category, titles in job_categories.items():
            for title in titles:
                all_job_titles.append(title)
        
        # Shuffle the list to avoid sequential searches for similar jobs
        random.shuffle(all_job_titles)
        
        # Process each job title
        for job_title in all_job_titles:
            print(f"\n{'='*50}")
            print(f"Processing search for: {job_title}")
            print(f"{'='*50}")
            
            # Search for the current job title
            if search_jobs(driver, job_title):
                page_num = 1
                jobs_on_current_search = []
                
                # Extract jobs from the first page
                print(f"Extracting jobs from page {page_num}...")
                page_jobs = extract_job_details(driver)
                jobs_on_current_search.extend(page_jobs)
                
                # Check for and navigate to next pages
                while check_for_next_page(driver):
                    page_num += 1
                    print(f"Extracting jobs from page {page_num}...")
                    page_jobs = extract_job_details(driver)
                    jobs_on_current_search.extend(page_jobs)
                    
                    # Add a random delay between page navigations (1-3 seconds)
                    time.sleep(random.uniform(1, 3))
                
                print(f"Found {len(jobs_on_current_search)} total jobs for '{job_title}'")
                all_jobs_data.extend(jobs_on_current_search)
                
                # Add a random delay between searches (2-5 seconds)
                time.sleep(random.uniform(2, 5))
            
            # If we have a lot of jobs already, save intermediate results
            if len(all_jobs_data) > 500:
                save_to_csv(all_jobs_data, "usajobs_tech_jobs_intermediate.csv")
        
        # Print summary information
        print(f"\n{'='*50}")
        print(f"Scraping complete! Total jobs collected: {len(all_jobs_data)}")
        
        # Get counts by category
        job_counts = {}
        for job in all_jobs_data:
            search_term = job.get("search_term", "Unknown")
            if search_term in job_counts:
                job_counts[search_term] += 1
            else:
                job_counts[search_term] = 1
        
        print("\nJobs by search term:")
        for term, count in job_counts.items():
            print(f"  {term}: {count} jobs")
        
        # Save all collected data
        if all_jobs_data:
            filepath = save_to_csv(all_jobs_data)
            print(f"\nAll job data has been saved to {filepath}")
        else:
            print("\nNo job data was collected.")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        # Save whatever data we have so far
        if all_jobs_data:
            save_to_csv(all_jobs_data, "usajobs_tech_jobs_error_recovery.csv")
    finally:
        # Close the browser
        driver.quit()
        print("Browser closed.")

if __name__ == "__main__":
    main()