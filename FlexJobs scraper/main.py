import time
import csv
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# Define the job categories and titles to search for
JOB_CATEGORIES = {
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
        "UI/UX Designer", "Technical Writer", "Embedded Software Engineer",
        "IT Consultant", "IT Auditor"
    ]
}

class FlexJobsScraper:
    def __init__(self, headless=False, output_file="flexjobs_tech_jobs.csv"):
        """Initialize the FlexJobs scraper"""
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        
        # Set window size larger for visibility
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Disable HTTP/2 to avoid protocol errors
        self.chrome_options.add_argument("--disable-http2")
        
        # Disable browser cache
        self.chrome_options.add_argument("--disable-application-cache")
        self.chrome_options.add_argument("--disable-cache")
        
        # More robust user agent
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
        
        # Additional options to make connections more reliable
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--ignore-certificate-errors")
        
        self.driver = None
        self.output_file = output_file
        
        # Initialize CSV file with headers
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['title', 'url', 'agency', 'location', 'salary', 'closing_date', 'search_term'])
        
        # Track unique job IDs to avoid duplicates
        self.scraped_job_ids = set()
    
    def start_browser(self):
        """Start the Chrome browser"""
        self.driver = webdriver.Chrome(options=self.chrome_options)
        # Maximize window for better visibility
        self.driver.maximize_window()
        
        # Set page load timeout to avoid hanging
        self.driver.set_page_load_timeout(30)
        
        print("Browser started successfully")
    
    def close_browser(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("Browser closed")
    
    def navigate_to_search_page(self, keyword):
        """Navigate to the FlexJobs search page with the specified keyword"""
        search_keyword = keyword.replace(" ", "%20")
        url = f"https://www.flexjobs.com/search?searchkeyword={search_keyword}&usecLocation=true"
        
        print(f"Navigating to: {url}")
        
        # Clear cookies and cache before navigating to a new search
        self.driver.delete_all_cookies()
        
        try:
            self.driver.get(url)
            
            # Wait for job listings to load
            try:
                wait = WebDriverWait(self.driver, 15)
                wait.until(EC.presence_of_element_located((By.ID, "job-table-wrapper")))
                
                # Check for no results
                if self.check_no_results():
                    print(f"No results found for '{keyword}'")
                    return False
                
                # Extra wait to ensure all elements load
                time.sleep(2)
                return True
            except TimeoutException:
                print(f"Timed out waiting for job listings to load for '{keyword}'")
                
                # If we time out, try to get a screenshot for debugging
                try:
                    screenshot_file = f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.driver.save_screenshot(screenshot_file)
                    print(f"Error screenshot saved to {screenshot_file}")
                except Exception as ss_err:
                    print(f"Failed to save screenshot: {str(ss_err)}")
                
                # Try to reload the page
                print("Attempting to reload the page...")
                self.driver.refresh()
                time.sleep(5)
                
                # Check if reload worked
                try:
                    wait = WebDriverWait(self.driver, 15)
                    wait.until(EC.presence_of_element_located((By.ID, "job-table-wrapper")))
                    print("Page reload successful")
                    return True
                except:
                    print("Page reload failed")
                    return False
                
            except Exception as e:
                print(f"Error navigating to search page: {str(e)}")
                return False
        except Exception as e:
            print(f"Driver error loading URL: {str(e)}")
            
            # Re-initialize browser if we encounter a fatal error
            print("Restarting browser due to connection error...")
            self.close_browser()
            time.sleep(5)
            self.start_browser()
            time.sleep(2)
            
            # Try one more time with the new browser
            try:
                print(f"Retrying navigation to: {url}")
                self.driver.get(url)
                wait = WebDriverWait(self.driver, 15)
                wait.until(EC.presence_of_element_located((By.ID, "job-table-wrapper")))
                print("Second attempt successful")
                return True
            except Exception as retry_err:
                print(f"Second attempt also failed: {str(retry_err)}")
                return False
    
    def check_no_results(self):
        """Check if the search returned no results"""
        try:
            # Look for common "no results" indicators
            no_results_xpath = "//div[contains(text(), 'No jobs found') or contains(text(), 'No Results Found')]"
            no_results_elements = self.driver.find_elements(By.XPATH, no_results_xpath)
            
            if no_results_elements and any(el.is_displayed() for el in no_results_elements):
                return True
            
            # Also check if job table is empty
            job_container = self.driver.find_element(By.ID, "job-table-wrapper")
            job_cards = job_container.find_elements(By.CLASS_NAME, "sc-jv5lm6-0")
            if not job_cards:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking for no results: {str(e)}")
            return False
    
    def extract_job_details(self, job_card, search_keyword):
        """Extract job details and format them according to the required CSV structure"""
        try:
            # Extract job ID to track duplicates
            job_id = job_card.get_attribute("id")
            
            # Skip if we've already processed this job
            if job_id in self.scraped_job_ids:
                return None
            
            self.scraped_job_ids.add(job_id)
            
            # Extract title
            title_element = job_card.find_element(By.CLASS_NAME, "sc-jv5lm6-13")
            title = title_element.text
            url = title_element.get_attribute("href")
            
            # Extract company/agency (may not be available for all jobs)
            try:
                agency_element = job_card.find_element(By.CLASS_NAME, "company-name")
                agency = agency_element.text
            except NoSuchElementException:
                # Try alternative methods to find company name
                try:
                    # Sometimes company name might be in the description
                    description = job_card.find_element(By.ID, f"description-{job_id}")
                    desc_text = description.text
                    if "Company:" in desc_text:
                        agency = desc_text.split("Company:")[1].split("\n")[0].strip()
                    else:
                        agency = "Not specified"
                except:
                    agency = "Not specified"
            
            # Extract location
            try:
                location_element = job_card.find_element(By.CSS_SELECTOR, ".allowed-location")
                location = location_element.text
            except NoSuchElementException:
                location = "Not specified"
            
            # Extract salary (usually in job tags or description)
            salary = "Not specified"
            
            # Check job tags for salary information
            tag_elements = job_card.find_elements(By.CLASS_NAME, "ifOAWE")
            for tag in tag_elements:
                tag_text = tag.text.lower()
                if any(term in tag_text for term in ["salary", "usd", "annually", "hourly", "$"]):
                    salary = tag.text
                    break
            
            # Extract posted date (use as closing date)
            try:
                date_span = job_card.find_element(By.ID, f"date-diff-{job_id}")
                closing_date = date_span.text
            except NoSuchElementException:
                closing_date = "Not specified"
            
            # Format all fields according to the required CSV structure
            job_data = {
                'title': title,
                'url': url,
                'agency': agency,
                'location': location,
                'salary': salary,
                'closing_date': closing_date,
                'search_term': search_keyword
            }
            
            # Highlight the element to show visually which job is being processed
            self.driver.execute_script("arguments[0].style.border='3px solid red'", job_card)
            time.sleep(0.5)  # Keep highlight visible briefly
            self.driver.execute_script("arguments[0].style.border=''", job_card)
            
            print(f"Found job: {title} | {location} | {closing_date}")
            
            return job_data
            
        except StaleElementReferenceException:
            print(f"Stale element encountered, skipping job")
            return None
        except Exception as e:
            print(f"Error extracting job details: {str(e)}")
            return None
    
    def append_to_csv(self, job_data):
        """Append a job entry to the CSV file"""
        if not job_data:
            return
            
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                job_data['title'],
                job_data['url'],
                job_data['agency'],
                job_data['location'],
                job_data['salary'],
                job_data['closing_date'],
                job_data['search_term']
            ])
    
    def scrape_all_jobs_on_page(self, search_keyword):
        """Scrape all job listings from the current search page"""
        try:
            # Find all job cards within the job table wrapper
            job_container = self.driver.find_element(By.ID, "job-table-wrapper")
            job_cards = job_container.find_elements(By.CLASS_NAME, "sc-jv5lm6-0")
            
            if not job_cards:
                print("No job cards found on this page")
                return 0
                
            jobs_scraped = 0
            
            for i, job_card in enumerate(job_cards, 1):
                print(f"Scraping job {i} of {len(job_cards)} for keyword '{search_keyword}'...")
                
                # Scroll job into view for better element detection
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", job_card)
                time.sleep(0.5)  # Small delay to let any dynamic content load
                
                job_data = self.extract_job_details(job_card, search_keyword)
                if job_data:
                    self.append_to_csv(job_data)
                    jobs_scraped += 1
            
            print(f"Successfully scraped {jobs_scraped} new jobs for keyword '{search_keyword}' on this page")
            return jobs_scraped
            
        except Exception as e:
            print(f"Error scraping jobs on page: {str(e)}")
            return 0
    
    def check_and_navigate_pagination(self):
        """Check if there are more pages of results and navigate to the next page if available"""
        try:
            # Look for pagination element
            pagination = self.driver.find_elements(By.CSS_SELECTOR, ".pagination, #search-pagination")
            if not pagination:
                return False
            
            # Find the "Next" button
            next_buttons = self.driver.find_elements(By.CSS_SELECTOR, 
                                                   "a[aria-label='Next Page'], .pagination-next, li.next a")
            
            # If no explicit next button, look for the current page and the page after it
            if not next_buttons:
                # Try to find the "Next" text or arrow symbol
                next_links = self.driver.find_elements(By.XPATH, 
                                                    "//a[contains(text(), 'Next') or contains(., '→') or contains(., '»')]")
                if next_links:
                    next_buttons = next_links
            
            # If we found a next button that's not disabled
            for btn in next_buttons:
                btn_class = btn.get_attribute("class") or ""
                if "disabled" not in btn_class and btn.is_displayed():
                    print("Clicking Next page button...")
                    
                    # Highlight the button to make it visible in the UI
                    self.driver.execute_script("arguments[0].style.border='3px solid green'", btn)
                    time.sleep(1)  # Keep highlight visible briefly
                    
                    # Get current URL before clicking
                    current_url = self.driver.current_url
                    
                    try:
                        # Click the next button using JavaScript
                        self.driver.execute_script("arguments[0].click();", btn)
                    except:
                        print("JS click failed, trying direct click...")
                        btn.click()
                    
                    # Wait for the new page to load
                    WebDriverWait(self.driver, 15).until(
                        lambda driver: driver.current_url != current_url
                    )
                    
                    # Additional wait for the content to load
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.ID, "job-table-wrapper"))
                    )
                    
                    time.sleep(2)  # Wait for content to fully load
                    return True
                
            return False
            
        except Exception as e:
            print(f"Error navigating pagination: {str(e)}")
            return False
    
    def scrape_jobs_for_search_term(self, search_keyword):
        """Scrape all jobs for a specific search term, including pagination"""
        if self.navigate_to_search_page(search_keyword):
            print(f"Successfully navigated to search page for '{search_keyword}'")
            
            page_num = 1
            total_jobs = 0
            
            # Process first page
            print(f"Scraping page {page_num}...")
            jobs_on_page = self.scrape_all_jobs_on_page(search_keyword)
            total_jobs += jobs_on_page
            
            # Process subsequent pages if available
            while self.check_and_navigate_pagination():
                page_num += 1
                print(f"Navigated to page {page_num}, scraping...")
                jobs_on_page = self.scrape_all_jobs_on_page(search_keyword)
                total_jobs += jobs_on_page
            
            print(f"Completed scraping for '{search_keyword}'. Found {total_jobs} jobs across {page_num} pages.")
            return total_jobs
        else:
            print(f"Failed to navigate to search page for '{search_keyword}' or no results found")
            return 0
    
    def run_full_scrape(self):
        """Scrape all job categories and titles"""
        total_jobs_scraped = 0
        search_terms = []
        
        # Flatten the job titles for easier iteration
        for category, titles in JOB_CATEGORIES.items():
            for title in titles:
                search_terms.append((category, title))
        
        # Count total search terms for progress tracking
        total_searches = len(search_terms)
        
        print(f"Starting scraping for {total_searches} different job searches")
        for idx, (category, title) in enumerate(search_terms, 1):
            print(f"\n[{idx}/{total_searches}] Category: {category} | Job: {title}")
            print("-" * 60)
            
            jobs_found = self.scrape_jobs_for_search_term(title)
            total_jobs_scraped += jobs_found
            
            print(f"Progress: {idx}/{total_searches} searches completed")
            
            # Add a longer delay between searches to be nice to the server
            # and avoid connection issues
            if idx < total_searches:
                delay = 10  # increased delay to 10 seconds
                print(f"Waiting {delay} seconds before next search...")
                time.sleep(delay)
        
        print(f"\nScraping completed. Total unique jobs found: {len(self.scraped_job_ids)}")
        print(f"Data saved to: {self.output_file}")
        
        return total_jobs_scraped

def main():
    # Create output file with timestamp to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"flexjobs_tech_jobs_{timestamp}.csv"
    
    # Set headless=False to see the browser in action
    scraper = FlexJobsScraper(headless=False, output_file=output_file)
    
    try:
        print(f"Starting FlexJobs scraper, output will be saved to {output_file}")
        start_time = time.time()
        
        scraper.start_browser()
        scraper.run_full_scrape()
        
        # Calculate and display elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal scraping time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    except Exception as e:
        print(f"An error occurred during scraping: {str(e)}")
    
    finally:
        # Ask user before closing the browser
        input("Press Enter to close the browser and finish the scraping process...")
        scraper.close_browser()
        print("Scraper shut down")

if __name__ == "__main__":
    main()