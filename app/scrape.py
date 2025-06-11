# import requests
# from bs4 import BeautifulSoup
# import time
# import random
# import re
# import csv

# def scrape_static_insurance_prices(url, insurance_type):
#     """
#     Scrapes a specific insurance price from a static landing page.

#     Args:
#         url (str): The URL of the static landing page.
#         insurance_type (str): 'general_liability' or 'workers_compensation'
#                               (used for identifying which price to look for).

#     Returns:
#         dict: A dictionary with 'url', 'insurance_type', and 'price',
#               or None if the price isn't found or an error occurs.
#     """
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }

#     try:
#         print(f"Fetching {url} for {insurance_type}...")
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching {url}: {e}")
#         return None

#     soup = BeautifulSoup(response.text, 'html.parser')
#     price = None

#     # --- CUSTOMIZE THESE SELECTORS BASED ON ACTUAL WEBSITE HTML ---
#     if insurance_type == 'general_liability':
#         # Example 1: Price inside a span with a specific class within a GL section
#         gl_section = soup.find('div', {'class': 'tableWrap_k1scE'}) # Adjust ID
#         if gl_section:
#             price_element = gl_section.find('td') # Adjust class
#             if price_element:
#                 price = price_element.get_text()
        
#         # Example 2: Price directly in a p tag with a specific text
#         if not price: # If not found by example 1, try example 2
#             price_element = soup.find('p', string=re.compile(r'General Liability starting from'))
#             if price_element:
#                 # Assuming the price is right after "from $"
#                 match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', price_element.get_text())
#                 if match:
#                     price = match.group(1)

#     elif insurance_type == 'workers_compensation':
#         # Example 1: Price inside a specific div with a WC class
#         wc_div = soup.find('div', class_='workers-comp-quote-box') # Adjust class
#         if wc_div:
#             price_element = wc_div.find('strong', class_='monthly-cost') # Adjust tag and class
#             if price_element:
#                 price = price_element.get_text()
        
#         # Example 2: Price in a data attribute
#         if not price:
#             price_element = soup.find('div', {'data-insurance-type': 'workers-compensation'})
#             if price_element and 'data-price' in price_element.attrs:
#                 price = price_element['data-price']

#     # --- END CUSTOMIZATION ---

#     if price:
#         # Clean the price string (remove $, commas, whitespace) and convert to float
#         cleaned_price = float(re.sub(r'[$,\s]', '', price))
#         print(f"Found price for {insurance_type}: ${cleaned_price:.2f}")
#         return {'url': url, 'insurance_type': insurance_type, 'price': cleaned_price}
#     else:
#         print(f"No price found for {insurance_type} on {url}")
#         return None

# if __name__ == "__main__":
#     # Define your list of URLs to scrape
#     # IMPORTANT: Replace these with actual URLs of static pages you find
#     # and adjust the selectors in the scrape_static_insurance_prices function accordingly.
#     target_pages = [
#         {"url": "https://www.thehartford.com/general-liability-insurance", "type": "general_liability"},
#         {"url": "https://www.progressivecommercial.com/business-insurance/general-liability-insurance/", "type": "general_liability"},
#         {"url": "https://www.geico.com/workers-compensation-insurance/", "type": "workers_compensation"},
#         # Add more URLs as you discover static price points
#         {"url": "https://www.insureon.com/small-business-insurance/general-liability/cost", "type": "general_liability"},
#         # {"url": "https://www.another-insurer.com/wc-average-costs", "type": "workers_compensation"},
#     ]

#     all_scraped_data = []

#     for page_info in target_pages:
#         scraped_data = scrape_static_insurance_prices(page_info['url'], page_info['type'])
#         if scraped_data:
#             all_scraped_data.append(scraped_data)

#         # Be polite: introduce a random delay between requests
#         delay = random.uniform(3, 8) # Random delay between 3 and 8 seconds
#         print(f"Waiting for {delay:.2f} seconds...")
#         time.sleep(delay)

#     # Output the results
#     if all_scraped_data:
#         print("\n--- All Scraped Prices ---")
#         for data in all_scraped_data:
#             print(f"Type: {data['insurance_type'].replace('_', ' ').title()}, Price: ${data['price']:.2f}, URL: {data['url']}")

#         # Optionally save to a CSV file
#         csv_file = 'us_insurance_prices.csv'
#         with open(csv_file, 'w', newline='', encoding='utf-8') as f:
#             fieldnames = ['url', 'insurance_type', 'price']
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(all_scraped_data)
#         print(f"\nScraped data saved to {csv_file}")
#     else:
#         print("\nNo prices were successfully scraped.")
from flask import Flask, jsonify, request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import csv # Not directly used in API response, but useful for understanding the scraping output format
import time
import random
import os # For checking driver path

app = Flask(__name__)

# --- Configuration ---
# IMPORTANT: Set this to the actual path of your chromedriver.exe (or geckodriver.exe)
# If chromedriver is in your system's PATH, you can leave driver_path=None
# Example: DRIVER_PATH = "C:/webdrivers/chromedriver.exe"
# Example: DRIVER_PATH = "/usr/local/bin/chromedriver"
DRIVER_PATH = None # Set this if your driver is not in PATH

# Ensure the driver exists if a custom path is provided
if DRIVER_PATH and not os.path.exists(DRIVER_PATH):
    raise FileNotFoundError(f"ChromeDriver not found at: {DRIVER_PATH}. Please check the path.")

# Define the sites to scrape with their respective URLs and names
SITES_TO_SCRAPE = [
    {'name': 'Insureon GL', 'url': 'https://www.insureon.com/small-business-insurance/general-liability/cost'},
    {'name': 'Forbes WC', 'url': 'https://www.forbes.com/advisor/business-insurance/workers-compensation-insurance-cost/'},
    #{'name': 'TechInsurance WC', 'url': 'https://www.techinsurance.com/workers-compensation-insurance/cost'}
]

# --- Helper function for cleaning price strings ---
def clean_price_string(price_text):
    """Removes currency symbols, commas, and whitespace, then converts to float."""
    if price_text is None:
        return None
    match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', price_text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            print(f"Warning: Could not convert '{match.group(1)}' to float.")
            return None
    return None

# --- Selenium Driver Initialization (called per request, be mindful of resources) ---
def get_selenium_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')

    if DRIVER_PATH:
        service = Service(DRIVER_PATH)
        return webdriver.Chrome(service=service, options=options)
    else:
        return webdriver.Chrome(options=options)

# --- Scraping function for Insureon ---
def scrape_insureon_gl_cost_by_state(driver, url):
    print(f"Scraping Insureon GL Costs from: {url}")
    scraped_data = []
    
    driver.get(url)
    try:
        table_wrap_selector = (By.CSS_SELECTOR, 'div[data-test-id="Table-Wrap"]')
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(table_wrap_selector)
        )
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        table_wrap_div = soup.find('div', {'data-test-id': 'Table-Wrap'})
        if table_wrap_div:
            cost_table = table_wrap_div.find('table', class_=['table_NoCqZ', 'firstColumnOverride_NoCqZ'])
            if cost_table:
                tbody = cost_table.find('tbody')
                if tbody:
                    rows = tbody.find_all('tr')
                    for row in rows:
                        state_th = row.find('th')
                        cost_td = row.find('td')

                        state = None
                        cost = None
                        
                        if state_th:
                            state_p = state_th.find('p')
                            if state_p:
                                state = state_p.get_text(strip=True)
                        
                        if cost_td:
                            cost_p = cost_td.find('p')
                            if cost_p:
                                cost = clean_price_string(cost_p.get_text(strip=True))
                        
                        if state and cost is not None:
                            scraped_data.append({
                                'source': 'Insureon',
                                'insurance_type': 'General Liability',
                                'metric': 'Cost per Month',
                                'state': state,
                                'cost': cost,
                                'url': url
                            })
            else:
                print("Insureon: Cost table not found inside the wrapper div with expected classes.")
        else:
            print("Insureon: 'div' with data-test-id='Table-Wrap' not found in Selenium's page source.")
    except Exception as e:
        print(f"Error scraping Insureon: {e}")
    return scraped_data

# --- Scraping function for Forbes Advisor ---
def scrape_forbes_wc_cost(driver, url):
    print(f"Scraping Forbes Advisor WC Costs from: {url}")
    scraped_data = []
    
    driver.get(url)
    try:
        # Wait for the tbody of the first table to be present
        table_container_selector = (By.CSS_SELECTOR, 'div.body-table-wrapper table.table-simple tbody')
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(table_container_selector)
        )
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        # The following selectors are for the first table structure
        cost_table = soup.find('div', class_='body-table-wrapper').find('table', class_='table-simple')
        print(cost_table)
        if cost_table:
            header_cols = [] # Not strictly needed if we just grab last column, but good for debugging
            tbody = cost_table.find('tbody') # This should now reliably be found
            if tbody:
                rows = tbody.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    # The first table usually has State and Cost (e.g., 2 columns)
                    # Or State and multiple cost columns, where -1 might be the most recent.
                    if len(cols) > 1: 
                        state = cols[0].get_text(strip=True)
                        # Assuming cost is in the last column for the first table.
                        # If it's consistently the second column, cols[1] would be more direct.
                        latest_year_cost_text = cols[-1].get_text(strip=True)
                        latest_year_cost = clean_price_string(latest_year_cost_text)
                            
                        if state and latest_year_cost is not None:
                            scraped_data.append({
                                'source': 'Forbes Advisor',
                                'insurance_type': 'Workers Compensation',
                                # Adjusted metric for the first table (actual content may vary)
                                'metric': 'Median Cost per $100 Payroll (First Table)', 
                                'state': state,
                                'cost': latest_year_cost,
                                'url': url
                            })
                    else:
                        print(f"Forbes (First Table): Skipping row with unexpected number of columns: {len(cols)}")
            else:
                print("Forbes (First Table): Table body (tbody) not found in parsed HTML.")
        else:
            print("Forbes (First Table): Cost table (div.body-table-wrapper table.table-simple) not found in parsed HTML.")
    except Exception as e:
        print(f"Error scraping Forbes Advisor: {e}")
    return scraped_data

# --- Scraping function for TechInsurance (average cost) ---
def scrape_techinsurance_wc_average_cost(driver, url):
    print(f"Scraping TechInsurance WC Average Cost from: {url}")
    scraped_data = []
    
    driver.get(url)
    try:
        article_text_div_selector = (By.CLASS_NAME, "article__text") 
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(article_text_div_selector)
        )
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        article_text_div = soup.find('div', class_='article__text')
        
        if article_text_div:
            average_cost_p = article_text_div.find('p', string=re.compile(r"On average, workers' compensation insurance costs"))
            
            if average_cost_p:
                cost_text = average_cost_p.get_text(strip=True)
                match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\sper month', cost_text)
                if match:
                    average_cost = clean_price_string(match.group(1))
                    if average_cost is not None:
                        scraped_data.append({
                            'source': 'TechInsurance',
                            'insurance_type': 'Workers Compensation',
                            'metric': 'Average Monthly Cost',
                            'state': 'National Average',
                            'cost': average_cost,
                            'url': url
                        })
                else:
                    print("TechInsurance: Average cost figure not found in expected format.")
            else:
                print("TechInsurance: Paragraph containing average cost not found within article text div.")
        else:
            print("TechInsurance: Article text div not found.")
            
    except Exception as e:
        print(f"Error scraping TechInsurance: {e}")
    return scraped_data

# --- Flask API Endpoint ---
@app.route('/scrape-insurance-prices', methods=['GET'])
def get_insurance_prices():
    all_scraped_data = []
    driver = None # Initialize driver to None

    try:
        driver = get_selenium_driver() # Get a new driver instance for this request

        for site_info in SITES_TO_SCRAPE:
            site_name = site_info['name']
            url = site_info['url']

            # Add a delay between sites to be polite
            delay = random.uniform(5, 10)
            print(f"API: Waiting {delay:.2f} seconds before scraping {site_name}...")
            time.sleep(delay)

            if site_name == 'Insureon GL':
                data = scrape_insureon_gl_cost_by_state(driver, url)
            elif site_name == 'Forbes WC':
                data = scrape_forbes_wc_cost(driver, url)
            elif site_name == 'TechInsurance WC':
                data = scrape_techinsurance_wc_average_cost(driver, url)
            else:
                print(f"API: Warning - No scraper defined for site: {site_name}")
                data = [] # Return empty list if site not recognized
            
            all_scraped_data.extend(data)
        
        return jsonify({
            'status': 'success',
            'data': all_scraped_data,
            'timestamp': time.time() # Add a timestamp for freshness
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': time.time()
        }), 500 # Internal Server Error
    finally:
        if driver:
            driver.quit() # Always quit the driver

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run: python your_script_name.py
    # Then open your browser and go to http://127.0.0.1:5000/scrape-insurance-prices
    # Or use curl: curl http://127.0.0.1:5000/scrape-insurance-prices
    print("Starting Flask API. Access at http://127.0.0.1:5000/scrape-insurance-prices")
    app.run(debug=True) # debug=True will restart server on code changes and show errors