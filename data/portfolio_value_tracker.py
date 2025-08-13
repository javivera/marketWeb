#!/usr/bin/env python3
"""
Portfolio Value Tracker
=======================
Reads a Solana wallet address and Tastytrade credentials from .env,
fetches portfolio values, and writes/appends a CSV with columns:
DATE,ARG,USA,CRYPTO,TOTAL,RETIRO/INGRESO,DIFF,DIF PCT,DIFF USA,DIFF USA PCT,DIFF CRYPTO,DIFF CRYPTO PCT

Notes:
- USA = Tastytrade portfolio value (sum across accounts)
- CRYPTO = Step Finance portfolio USD value via Selenium scraping
- RETIRO/INGRESO is left blank (user will fill later)
- DIFF columns are computed vs previous day's values in the CSV
- Requires selenium and webdriver-manager for crypto portfolio scraping
"""

import os
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional
import re
import time

# Selenium imports for web scraping
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    try:
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        SELENIUM_MANAGER_AVAILABLE = True
    except ImportError:
        Service = None
        ChromeDriverManager = None
        SELENIUM_MANAGER_AVAILABLE = False
    SELENIUM_AVAILABLE = True
except ImportError:
    # Selenium not available - will be handled gracefully in functions
    webdriver = None
    ChromeOptions = None
    By = None
    WebDriverWait = None
    EC = None
    Service = None
    ChromeDriverManager = None
    SELENIUM_AVAILABLE = False
    SELENIUM_MANAGER_AVAILABLE = False


DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / 'portfolio_values.csv'


def load_env_file(env_path: str = '.env') -> dict:
    """Load .env from common locations and merge them (later paths override)."""
    env_vars: dict = {}
    loaded_paths = []

    # Candidate paths: CWD .env, project root .env, data/.env, explicit env_path
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    candidates = [
        Path.cwd() / '.env',              # running working directory
        project_root / '.env',            # repo root
        script_path.parent / '.env',      # data/.env next to this script
    ]
    # If an explicit env_path was provided and isn't already in the list, append it last
    explicit = Path(env_path)
    if explicit not in candidates:
        candidates.append(explicit)

    for path in candidates:
        try:
            if path.exists():
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            env_vars[key] = value
                loaded_paths.append(str(path))
        except Exception:
            continue

    if loaded_paths:
        print(f"ℹ️  Loaded .env from: {', '.join(loaded_paths)}")
    else:
        print("⚠️  No .env files found in CWD, project root, or data/.env")

    return env_vars

def get_step_portfolio_selenium_usd(wallet_address: str, debug: bool = True, headless: bool = True, wait_seconds: float = 5.0) -> Optional[float]:
    """Use Selenium to render Step Finance portfolio and read the USD value from specific XPath.
    
    Target: //*[@id="layout-content-wrapper"]/main/div[1]/div[2]/span
    Requires 'selenium' (and optionally 'webdriver-manager'). If unavailable, returns None.
    
    NOTE: Runs in headless mode by default. Set headless=False for debugging.
    """
    if not SELENIUM_AVAILABLE:
        if debug:
            print("⚠️  Selenium not available - install with: pip install selenium")
        return None

    url = f"https://app.step.finance/portfolio?wallet={wallet_address}"
    target_xpath = '//*[@id="layout-content-wrapper"]/main/div[1]/div[2]/span'
    
    # Regex to extract dollar amounts
    money_re = re.compile(r'\$([0-9,]+(?:\.[0-9]{2})?)')

    def _norm(s: str) -> str:
        """Normalize text by removing non-breaking spaces and other Unicode"""
        return (s or '').replace('\xa0', ' ').replace('\u202f', ' ').strip()

    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        if SELENIUM_MANAGER_AVAILABLE and ChromeDriverManager is not None:
            if debug:
                print("Step-selenium: using webdriver-manager to install chromedriver")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        else:
            if debug:
                print("Step-selenium: using system chromedriver")
            driver = webdriver.Chrome(options=options)

        # Execute CDP command to remove webdriver property
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """
        })

        if debug:
            print(f"Step-selenium: navigating to {url}")
        driver.get(url)

        # Wait for document ready
        try:
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except Exception:
            pass

        # Wait for main layout container
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#layout-content-wrapper main'))
            )
            if debug:
                print("Step-selenium: main layout container found")
        except Exception:
            if debug:
                print("Step-selenium: main layout not detected within 30s")

        # Extra wait for SPA content to render
        if wait_seconds > 0:
            if debug:
                print(f"Step-selenium: waiting {wait_seconds}s for SPA content")
            time.sleep(wait_seconds)

        # Try to find the target element
        try:
            element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, target_xpath))
            )
            text = _norm(element.text)
            if debug:
                print(f"Step-selenium: found target element with text: '{text}'")
            
            # Extract dollar amount
            match = money_re.search(text)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                if debug:
                    print(f"Step-selenium: extracted amount: ${amount:.2f}")
                return amount
            else:
                if debug:
                    print(f"Step-selenium: no dollar amount found in text: '{text}'")
        except Exception as e:
            if debug:
                print(f"Step-selenium: target element not found: {e}")

        # Fallback: try to find any span with step-number class
        try:
            elements = driver.find_elements(By.XPATH, "//span[contains(@class,'step-number')]")
            if debug:
                print(f"Step-selenium: fallback found {len(elements)} step-number spans")
            
            for elem in elements:
                text = _norm(elem.text)
                match = money_re.search(text)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if amount > 10:  # Filter out small amounts
                        if debug:
                            print(f"Step-selenium: fallback extracted amount: ${amount:.2f}")
                        return amount
        except Exception as e:
            if debug:
                print(f"Step-selenium: fallback search failed: {e}")

        return None

    except Exception as e:
        if debug:
            print(f"Step-selenium: error {e}")
        return None
    finally:
        if driver is not None:
            if not headless and debug:
                # Give time to visually inspect when headed
                try:
                    time.sleep(3)
                except Exception:
                    pass
            try:
                driver.quit()
            except Exception:
                pass

def get_tastytrade_portfolio_value(username: str, password: str, debug: bool = True) -> Optional[float]:
    """Fetch total account value from TastyTrade using the tastytrade SDK"""
    try:
        from tastytrade import Session, Account
    except ImportError:
        if debug:
            print("⚠️  tastytrade library not installed. Install with: pip install tastytrade")
        return None
    
    try:
        if debug:
            print("🔐 Logging into TastyTrade...")
        session = Session(username, password)
        
        if debug:
            print("📊 Fetching account information...")
        
        # Use the correct API pattern
        accounts = Account.get(session)
        
        if not accounts or len(accounts) < 2:
            print("⚠️  No TastyTrade accounts found or insufficient accounts")
            return None
        
        # Use second account (index 1 as specified)
        account = accounts[1]
        
        if debug:
            print(f"✅ Found {len(accounts)} account(s)")
            # Try to get account identifier safely
            account_id = "Unknown"
            for attr in ['account_number', 'account-number', 'accountNumber', 'id', 'number']:
                if hasattr(account, attr):
                    account_id = getattr(account, attr)
                    break
            print(f"📈 Getting balances for account: {account_id}")
        
        # Get balances using the account object
        balances = account.get_balances(session)
        
        # AccountBalance object - access net_liquidating_value as attribute
        net_liquidating_value = balances.net_liquidating_value
        
        if debug:
            print(f"💰 TastyTrade net_liquidating_value = {net_liquidating_value} (type: {type(net_liquidating_value)})")
        
        if net_liquidating_value is not None:
            try:
                value = float(net_liquidating_value)
                print(f"✅ TastyTrade account value: ${value:.2f}")
                return value
            except (ValueError, TypeError) as e:
                print(f"⚠️  Could not convert net liquidating value to float: {net_liquidating_value} - {e}")
                return None
        else:
            print(f"⚠️  Net liquidating value is None")
            return None
            
    except Exception as e:
        print(f"⚠️  Error fetching TastyTrade portfolio: {e}")
        if debug:
            import traceback
            print(f"🔍 Full error traceback:")
            traceback.print_exc()
        return None


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read all rows from CSV file"""
    if not csv_path.exists():
        return []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"⚠️  Error reading CSV {csv_path}: {e}")
        return []


def _save_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Save all rows to CSV file"""
    if not rows:
        return
        
    try:
        # Ensure parent directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine fieldnames from existing data
        base_fieldnames = ['DATE', 'ARG', 'USA', 'CRYPTO', 'TOTAL', 'RETIRO/INGRESO', 'DIFF', 'DIF PCT', 'DIFF USA', 'DIFF USA PCT', 'DIFF CRYPTO', 'DIFF CRYPTO PCT']
        all_fieldnames = set(base_fieldnames)
        
        # Add any additional columns found in the data
        for row in rows:
            all_fieldnames.update(row.keys())
        
        # Convert to list and ensure base fields come first
        fieldnames = base_fieldnames + [f for f in sorted(all_fieldnames) if f not in base_fieldnames]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"⚠️  Error saving CSV {csv_path}: {e}")


def upsert_daily_row(csv_path: Path, new_row: dict[str, str]) -> None:
    """Insert or update daily portfolio row, calculating DIFF and DIF PCT"""
    rows = _read_rows(csv_path)
    
    # Find existing row for same date
    date = new_row['DATE']
    existing_idx = None
    for i, row in enumerate(rows):
        if row.get('DATE') == date:
            existing_idx = i
            break
    
    # Calculate DIFF and DIF PCT compared to previous day
    prev_row = None
    
    if existing_idx is not None:
        # Updating existing row - get the previous row if exists
        if existing_idx > 0:
            prev_row = rows[existing_idx - 1]
    else:
        # Adding new row - get the last row if exists
        if rows:
            prev_row = rows[-1]
    
    # Calculate differences
    if prev_row:
        try:
            current_total = float(new_row['TOTAL'])
            prev_total = float(prev_row.get('TOTAL', '0'))
            
            # Calculate USA difference
            current_usa = float(new_row.get('USA', '0') or '0')
            prev_usa = float(prev_row.get('USA', '0') or '0')
            usa_diff = current_usa - prev_usa
            usa_diff_pct = (usa_diff / prev_usa * 100) if prev_usa > 0 else 0
            
            # Calculate CRYPTO difference
            current_crypto = float(new_row.get('CRYPTO', '0') or '0')
            prev_crypto = float(prev_row.get('CRYPTO', '0') or '0')
            crypto_diff = current_crypto - prev_crypto
            crypto_diff_pct = (crypto_diff / prev_crypto * 100) if prev_crypto > 0 else 0
            
            if prev_total > 0:
                diff = current_total - prev_total
                diff_pct = (diff / prev_total) * 100
                new_row['DIFF'] = f"{diff:.2f}"
                new_row['DIF PCT'] = f"{diff_pct:.2f}%"
                print(f"💰 Calculated DIFF: ${diff:.2f} ({diff_pct:.2f}%)")
            else:
                new_row['DIFF'] = ''
                new_row['DIF PCT'] = ''
            
            # Set USA values
            new_row['DIFF USA'] = f"{usa_diff:.2f}"
            new_row['DIFF USA PCT'] = f"{usa_diff_pct:.2f}%" if prev_usa > 0 else ''
            
            # Set CRYPTO values
            new_row['DIFF CRYPTO'] = f"{crypto_diff:.2f}"
            new_row['DIFF CRYPTO PCT'] = f"{crypto_diff_pct:.2f}%" if prev_crypto > 0 else ''
            
            print(f"🇺🇸 USA DIFF: ${usa_diff:.2f} ({usa_diff_pct:.2f}%)")
            print(f"🪙 CRYPTO DIFF: ${crypto_diff:.2f} ({crypto_diff_pct:.2f}%)")
            
        except ValueError as e:
            print(f"⚠️  Error calculating DIFF: {e}")
            new_row['DIFF'] = ''
            new_row['DIF PCT'] = ''
            new_row['DIFF USA'] = ''
            new_row['DIFF USA PCT'] = ''
            new_row['DIFF CRYPTO'] = ''
            new_row['DIFF CRYPTO PCT'] = ''
    else:
        # First row - no previous data to compare
        new_row['DIFF'] = ''
        new_row['DIF PCT'] = ''
        new_row['DIFF USA'] = ''
        new_row['DIFF USA PCT'] = ''
        new_row['DIFF CRYPTO'] = ''
        new_row['DIFF CRYPTO PCT'] = ''
        print("ℹ️  First row - no previous data for DIFF calculation")
    
    # Update or append row
    if existing_idx is not None:
        rows[existing_idx] = new_row
        print(f"Updated existing row for {date}")
    else:
        rows.append(new_row)
        print(f"Added new row for {date}")
    
    # Sort by date
    rows.sort(key=lambda r: r.get('DATE', ''))
    
    _save_rows(csv_path, rows)


def recalculate_all_diffs(csv_path: Path) -> None:
    """Recalculate DIFF and DIF PCT for all rows in the CSV"""
    rows = _read_rows(csv_path)
    if not rows:
        print("No data to recalculate")
        return
    
    # Sort by date to ensure proper order
    rows.sort(key=lambda r: r.get('DATE', ''))
    
    for i, row in enumerate(rows):
        if i == 0:
            # First row - no previous data
            row['DIFF'] = ''
            row['DIF PCT'] = ''
            row['DIFF USA'] = ''
            row['DIFF USA PCT'] = ''
            row['DIFF CRYPTO'] = ''
            row['DIFF CRYPTO PCT'] = ''
        else:
            # Calculate against previous row
            try:
                current_total = float(row['TOTAL'])
                prev_total = float(rows[i-1].get('TOTAL', '0'))
                
                # Calculate USA difference
                current_usa = float(row.get('USA', '0') or '0')
                prev_usa = float(rows[i-1].get('USA', '0') or '0')
                usa_diff = current_usa - prev_usa
                usa_diff_pct = (usa_diff / prev_usa * 100) if prev_usa > 0 else 0
                
                # Calculate CRYPTO difference
                current_crypto = float(row.get('CRYPTO', '0') or '0')
                prev_crypto = float(rows[i-1].get('CRYPTO', '0') or '0')
                crypto_diff = current_crypto - prev_crypto
                crypto_diff_pct = (crypto_diff / prev_crypto * 100) if prev_crypto > 0 else 0
                
                if prev_total > 0:
                    diff = current_total - prev_total
                    diff_pct = (diff / prev_total) * 100
                    row['DIFF'] = f"{diff:.2f}"
                    row['DIF PCT'] = f"{diff_pct:.2f}%"
                else:
                    row['DIFF'] = ''
                    row['DIF PCT'] = ''
                
                row['DIFF USA'] = f"{usa_diff:.2f}"
                row['DIFF USA PCT'] = f"{usa_diff_pct:.2f}%" if prev_usa > 0 else ''
                row['DIFF CRYPTO'] = f"{crypto_diff:.2f}"
                row['DIFF CRYPTO PCT'] = f"{crypto_diff_pct:.2f}%" if prev_crypto > 0 else ''
                
            except ValueError:
                row['DIFF'] = ''
                row['DIF PCT'] = ''
                row['DIFF USA'] = ''
                row['DIFF USA PCT'] = ''
                row['DIFF CRYPTO'] = ''
                row['DIFF CRYPTO PCT'] = ''
    
    _save_rows(csv_path, rows)
    print(f"✅ Recalculated DIFF and DIF PCT for {len(rows)} rows")


def get_solana_portfolio_usd(wallet_address: str) -> Optional[float]:
    """Fallback: Get SOL balance and convert to USD using CoinGecko API"""
    try:
        import requests
    except ImportError:
        print("⚠️  requests library not installed for SOL fallback")
        return None
    
    try:
        # Get SOL balance
        rpc_url = "https://api.mainnet-beta.solana.com"
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [wallet_address]
        }
        
        response = requests.post(rpc_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if 'result' in result and 'value' in result['result']:
            balance_lamports = result['result']['value']
            balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        else:
            print("⚠️  Invalid SOL balance response")
            return None
        
        # Get SOL price
        price_response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
            timeout=10
        )
        price_response.raise_for_status()
        
        price_data = price_response.json()
        sol_price = price_data.get('solana', {}).get('usd', 0)
        
        if sol_price > 0:
            total_usd = balance_sol * sol_price
            print(f"✅ SOL balance: {balance_sol:.4f} SOL @ ${sol_price:.2f} = ${total_usd:.2f}")
            return total_usd
        else:
            print("⚠️  Unable to get SOL price")
            return None
            
    except Exception as e:
        print(f"⚠️  Error fetching SOL portfolio: {e}")
        return None


def main():
    import sys
    
    # Check for recalculate command
    if len(sys.argv) > 1 and sys.argv[1] == 'recalculate':
        print("🔄 Recalculating DIFF and DIF PCT for all rows...")
        recalculate_all_diffs(CSV_PATH)
        return
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config = load_env_file()
    sol_addr = config.get('SOLANA_WALLET') or config.get('SOL_WALLET') or ''
    tasty_user = config.get('TASTY_USER') or config.get('TASTY_USERNAME') or ''
    tasty_pass = config.get('TASTY_PASS') or config.get('TASTY_PASSWORD') or ''
    # Default to headless browser unless explicitly set to show
    show_browser_env = config.get('STEP_SHOW_BROWSER', '').strip().lower()
    show_browser = show_browser_env in {'1', 'true', 'yes', 'on'}  # Default to False unless explicitly enabled
    wait_seconds = float(config.get('STEP_WAIT_SEC') or '5')  # Default to 5 seconds
    
    # Get manual values from environment
    retiro_ingreso = config.get('RETIRO_INGRESO', '').strip()
    custom_column = config.get('CUSTOM_COLUMN', '').strip()
    arg_column = config.get('ARG', '').strip()

    if not sol_addr:
        print('⚠️  SOLANA_WALLET not set in .env; CRYPTO value may be empty.')
    if not tasty_user or not tasty_pass:
        print('⚠️  TASTY_USER/TASTY_PASS not set in .env; USA value may be empty.')

    # Fetch values using Selenium for Step Finance portfolio
    crypto_val = None
    if sol_addr:
        crypto_val = get_step_portfolio_selenium_usd(sol_addr, debug=True, headless=not show_browser, wait_seconds=wait_seconds)
        if crypto_val is None:
            print("ℹ️  Step Finance scraping failed; falling back to SOL-only valuation.")
            crypto_val = get_solana_portfolio_usd(sol_addr)
        else:
            print(f"✅ Step portfolio USD total: {crypto_val:.2f}")
    usa_val = get_tastytrade_portfolio_value(tasty_user, tasty_pass, debug=True) if (tasty_user and tasty_pass) else None

    # Default missing values to 0 for totals
    crypto_val_num = float(crypto_val) if crypto_val is not None else 0.0
    usa_val_num = float(usa_val) if usa_val is not None else 0.0
    arg_val_num = float(arg_column) if arg_column and arg_column.replace('.', '').replace('-', '').isdigit() else 0.0
    total = usa_val_num + crypto_val_num + arg_val_num

    # Build daily row (DATE only)
    now_str = datetime.now().strftime('%Y-%m-%d')
    row = {
        'DATE': now_str,
        'ARG': arg_column,  # from environment variable - first column after date
        'USA': f"{usa_val_num:.2f}" if usa_val is not None else '',
        'CRYPTO': f"{crypto_val_num:.2f}" if crypto_val is not None else '',
        'TOTAL': f"{total:.2f}",
        'RETIRO/INGRESO': retiro_ingreso,  # from environment variable
        'DIFF': '',
        'DIF PCT': '',
        'DIFF USA': '',
        'DIFF USA PCT': '',
        'DIFF CRYPTO': '',
        'DIFF CRYPTO PCT': '',
    }
    
    # Add custom column if specified
    if custom_column:
        row['CUSTOM'] = custom_column

    upsert_daily_row(CSV_PATH, row)
    print(f"✅ Upserted daily row in {CSV_PATH}")


if __name__ == '__main__':
    main()
