# app.py
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from bs4 import BeautifulSoup
import os

app = Flask(__name__)

# --- WebDriver Initialization Helper ---
def get_webdriver() -> webdriver.Chrome:
    """
    Initializes and returns a Chrome WebDriver instance configured for headless operation.
    Uses webdriver_manager to automatically handle ChromeDriver download/management.
    """
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.maximize_window()
    print("ChromeDriver initialized successfully via webdriver_manager.")
    return driver

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

# --- Thimble Scraping Logic ---
def scrape_thimble_insurance_coverage(
    num_employees: int,
    profession_text: str,
    zip_code: str,
    equipment_value: float = 0.0,
    coverage_limit_value: int = 1 # 1 for $1M, 2 for $2M
) -> dict:
    """
    Scrapes "Recommended coverage" from thimble.com's insurance calculator using Selenium.
    """
    driver = None
    try:
        driver = get_webdriver()
        calculator_url = "https://www.thimble.com/insurance-calculator/"
        print(f"Thimble: Navigating to: {calculator_url}")
        driver.get(calculator_url)

        wait = WebDriverWait(driver, 20)

        # 1. Fill 'Number of Employees'
        try:
            employees_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Number of Employees')]/ancestor::div[@class='smb-form-item']//input[@class='smb-input__inner']")))
            employees_input.clear()
            employees_input.send_keys(str(num_employees))
            employees_input.send_keys(Keys.TAB)
            time.sleep(0.5)
            print(f"Thimble: Filled Number of Employees: {num_employees}")
        except Exception as e:
            print(f"Thimble: Error interacting with 'Number of Employees': {e}")
            return {"status": "error", "message": f"Failed to fill employees on Thimble: {e}"}

        # 2. Fill 'Profession'
        try:
            profession_input_field = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Handymen' and @class='smb-input__inner smb-input__inner--has-suffix']")))
            
            profession_input_field.click() 
            print("Thimble: Clicked profession input field to open dropdown.")
            
            profession_input_field.send_keys(profession_text)
            print(f"Thimble: Typed profession: {profession_text}")

            profession_option_xpath = f"//div[@id='smbPoperRoot']//div[@class='insurance-calculator-widget__pop-option'][contains(text(), '{profession_text}')]"
            profession_option = wait.until(EC.element_to_be_clickable((By.XPATH, profession_option_xpath)))
            
            profession_option.click()
            print(f"Thimble: Selected profession '{profession_text}' from dropdown.")
            time.sleep(1)
        except Exception as e:
            print(f"Thimble: Error interacting with 'Profession': {e}")
            return {"status": "error", "message": f"Failed to fill profession on Thimble: {e}"}

        # 3. Fill 'ZIP Code'
        try:
            zip_code_input_field = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='90402' and @class='smb-input__inner smb-input__inner--has-suffix']")))
            
            zip_code_input_field.click()
            print("Thimble: Clicked ZIP Code input field.")
            
            zip_code_input_field.clear()
            zip_code_input_field.send_keys(zip_code)
            zip_code_input_field.send_keys(Keys.RETURN)
            time.sleep(1)
            print(f"Thimble: Filled ZIP Code: {zip_code}")
        except Exception as e:
            print(f"Thimble: Error interacting with 'ZIP Code': {e}")
            return {"status": "error", "message": f"Failed to fill ZIP code on Thimble: {e}"}

        # 4. Fill 'Value of Your Equipment' (Optional)
        if equipment_value > 0:
            try:
                equipment_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Value of Your Equipment')]/ancestor::div[@class='smb-form-item']//input[@class='smb-input__inner']")))
                equipment_input.clear()
                equipment_input.send_keys(str(int(equipment_value)))
                equipment_input.send_keys(Keys.TAB)
                time.sleep(0.5)
                print(f"Thimble: Filled Equipment Value: ${equipment_value:,.2f}")
            except Exception as e:
                print(f"Thimble: Warning: Could not interact with 'Equipment Value' input: {e}")

        # # 5. Select 'Coverage Limit'
        # try:
        #     limit_text = f"${coverage_limit_value}M"
        #     limit_option = wait.until(EC.element_to_be_clickable((By.XPATH, f"//div[@class='smb-slider__label-item']//span[contains(text(), '{limit_text}')]/ancestor::div[@class='smb-slider__label-item']")))
        #     limit_option.click()
        #     time.sleep(0.5)
        #     print(f"Thimble: Selected Coverage Limit: {limit_text}")
        # except Exception as e:
        #     print(f"Thimble: Error interacting with 'Coverage Limit': {e}")
        #     return {"status": "error", "message": f"Failed to select coverage limit on Thimble: {e}"}

        time.sleep(3) 

        # 6. Extract "Recommended coverage" result
        try:
            price_span = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'insurance-calculator-widget__price')]//span[@class='smb-typoel smb-typo-h2']")))
            avg_month_span = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'insurance-calculator-widget__price')]//span[@class='smb-typoel smb-typo-body' and contains(text(), 'Avg / Month')]")))
            
            price_value = price_span.get_attribute('textContent').strip()
            coverage_text = f"{price_value} {avg_month_span.get_attribute('textContent').strip()}"
            
            print(f"Thimble: Scraped 'Recommended coverage': {coverage_text}")
            return {"status": "success", "recommended_coverage": coverage_text}
        except Exception as e:
            print(f"Thimble: Error extracting price: {e}")
            return {"status": "error", "message": f"Failed to extract recommended coverage on Thimble: {e}"}

    except Exception as e:
        print(f"Thimble: An overall error occurred during scraping: {e}")
        return {"status": "error", "message": f"An unexpected error occurred on Thimble: {e}"}
    finally:
        if driver:
            driver.quit() # Ensure the browser is closed even if an error occurs

# --- Scraping function for Insureon ---
def scrape_insureon_gl_cost_by_state(state: str):
    url = "https://www.insureon.com/small-business-insurance/general-liability/cost"
    print(f"Scraping Insureon GL Costs from: {url}")

    driver = get_webdriver()
    driver.get(url)
    state_input = state.lower()

    # wait = WebDriverWait(driver, 20)
    scraped_data = { "status": "failed" }
    
    # driver.get(url)
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
                            if state.lower() == state_input.lower():
                                scraped_data = {
                                    'recommended_coverage': f"${cost}/month",
                                    "status": "success"
                                }
            else:
                print("Insureon: Cost table not found inside the wrapper div with expected classes.")
        else:
            print("Insureon: 'div' with data-test-id='Table-Wrap' not found in Selenium's page source.")
    except Exception as e:
        print(f"Error scraping Insureon: {e}")
    return scraped_data

# --- NextInsurance Scraping Logic ---
def scrape_nextinsurance_coverage(
    state: str,
    industry: str
) -> dict:
    """
    Scrapes the price from nextinsurance.com's general liability calculator using Selenium.
    """
    driver = None
    try:
        driver = get_webdriver()
        calculator_url = "https://www.nextinsurance.com/general-liability-insurance/calculator/"
        print(f"NextInsurance: Navigating to: {calculator_url}")
        driver.get(calculator_url)

        wait = WebDriverWait(driver, 20)

        # 1. Fill 'Where is your business located? (State)'
        try:
            state_input = wait.until(EC.element_to_be_clickable((By.ID, "price-calculator-state")))
            state_input.click()
            state_input.clear()
            state_input.send_keys(state)
            print(f"NextInsurance: Typed state: {state}")
            
            state_option_xpath = f"//div[contains(@class, 'state-dropdown')]//*[contains(@class, 'css-') and contains(text(), '{state}')]"
            state_option = wait.until(EC.element_to_be_clickable((By.XPATH, state_option_xpath)))
            state_option.click()
            print(f"NextInsurance: Clicked state option: {state}")
            time.sleep(1)
        except Exception as e:
            print(f"NextInsurance: Error interacting with 'State': {e}")
            return {"status": "error", "message": f"Failed to fill state on NextInsurance: {e}"}

        # 2. Fill 'What is your industry?'
        try:
            industry_input = wait.until(EC.element_to_be_clickable((By.ID, "algolia-price-calculator")))
            industry_input.click()
            industry_input.clear()
            industry_input.send_keys(industry)
            print(f"NextInsurance: Typed industry: {industry}")
            
            time.sleep(1.5)
            
            industry_option_xpath = (
                f"//div[@data-testid='drop-down-options']"
                f"//div[contains(@data-testid, 'dropdown-item-cob-input') and contains(text(), '{industry}')]"
            )
            
            industry_option = wait.until(EC.element_to_be_clickable((By.XPATH, industry_option_xpath)))
            industry_option.click()
            print(f"NextInsurance: Clicked industry option: {industry}")
            time.sleep(1)
        except Exception as e:
            print(f"NextInsurance: Error interacting with 'Industry': {e}")
            return {"status": "error", "message": f"Failed to fill industry on NextInsurance: {e}"}
        
        time.sleep(3) 

        # 3. Scrape the price from "data-cy="price-calculator-calculation-field-value""
        try:
            price_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-cy='price-calculator-calculation-field-value']")))
            price_text = price_element.get_attribute('textContent').strip()
            print(f"NextInsurance: Scraped price: {price_text}")
            return {"status": "success", "recommended_coverage": price_text}
        except Exception as e:
            print(f"NextInsurance: Error extracting price: {e}")
            return {"status": "error", "message": f"Failed to extract price from NextInsurance: {e}"}

    except Exception as e:
        print(f"NextInsurance: An overall error occurred during scraping: {e}")
        return {"status": "error", "message": f"An unexpected error occurred on NextInsurance: {e}"}
    finally:
        if driver:
            driver.quit()

# --- Generic Flask Route ---
@app.route('/scrape-insurance', methods=['POST'])
def scrape_insurance():
    """
    Generic API endpoint to scrape insurance quotes from different providers.
    Expects a JSON payload that can include parameters for one or more providers.
    
    Payload Structure:
    {
        "thimble_params": { ... },
        "nextinsurance_params": { ... },
        "insureon_params": { ... }
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    response_data = {}
    
    # --- Handle Thimble Request ---
    thimble_params = data.get('thimble_params')
    if thimble_params:
        print("\nAttempting Thimble scrape...")
        num_employees = thimble_params.get('num_employees')
        profession_text = thimble_params.get('profession_text')
        zip_code = thimble_params.get('zip_code')
        equipment_value = thimble_params.get('equipment_value', 0.0)
        coverage_limit_value = thimble_params.get('coverage_limit_value', 1)

        if not all([num_employees, profession_text, zip_code]):
            thimble_result = {"status": "error", "message": "Missing required parameters for Thimble: num_employees, profession_text, zip_code"}
        else:
            try:
                num_employees = int(num_employees)
                equipment_value = float(equipment_value)
                coverage_limit_value = int(coverage_limit_value)
                thimble_result = scrape_thimble_insurance_coverage(
                    num_employees=num_employees,
                    profession_text=profession_text,
                    zip_code=zip_code,
                    equipment_value=equipment_value,
                    coverage_limit_value=coverage_limit_value
                )
            except ValueError:
                thimble_result = {"status": "error", "message": "Invalid data types for numeric fields for Thimble"}
        response_data['thimble_result'] = thimble_result

    # --- Handle NextInsurance Request ---
    nextinsurance_params = data.get('nextinsurance_params')
    if nextinsurance_params:
        print("\nAttempting NextInsurance scrape...")
        state = nextinsurance_params.get('state')
        industry = nextinsurance_params.get('industry')

        if not all([state, industry]):
            nextinsurance_result = {"status": "error", "message": "Missing required parameters for NextInsurance: state, industry"}
        else:
            nextinsurance_result = scrape_nextinsurance_coverage(
                state=state,
                industry=industry
            )
        response_data['nextinsurance_result'] = nextinsurance_result
    
    insureon_params = data.get('insureon_params')
    print(f"Insureon params: {insureon_params}")
    if insureon_params:
        print("\nAttempting Insureon scrape...")
        state = insureon_params.get('state')
        print(f"Insureon state: {state}")
        response_data['insureon_result'] = scrape_insureon_gl_cost_by_state(state)

    # If no provider's parameters were provided
    if not response_data:
        return jsonify({"status": "error", "message": "No valid provider parameters found in the request. Please provide 'thimble_params' or 'nextinsurance_params'."}), 400

    # Determine overall HTTP status: 200 if at least one scrape was successful, else 500
    overall_status = 200
    if thimble_params and response_data.get('thimble_result', {}).get('status') == 'error':
        overall_status = 500
    if nextinsurance_params and response_data.get('nextinsurance_result', {}).get('status') == 'error':
        overall_status = 500
    
    return jsonify(response_data), overall_status


if __name__ == '__main__':
    app.run(debug=True, port=5000)