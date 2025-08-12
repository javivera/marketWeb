import os
import sys
import asyncio
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Third party libs already used in repo
from dotenv import load_dotenv
from tastytrade import Session

# yfinance only as graceful fallback if TastyTrade historical endpoint fails
import yfinance as yf

load_dotenv()

################################################################################
# Assumptions / Notes
# 1. TastyTrade Python SDK (tastytrade) at time of writing does not expose a
#    documented high-level helper in this repo for historical EOD prices.
#    We attempt to call the underlying REST price history endpoint directly.
#    Endpoint pattern (assumed):
#       https://api.tastytrade.com/instruments/equities/{symbol}/price-history
#    with query params: start-date, end-date, interval=1d
# 2. If that request fails for ANY symbol, we fall back to yfinance for that
#    symbol only (so we can still build a correlation matrix).
# 3. Correlation is computed on DAILY RETURNS (pct change) to be more meaningful
#    than price correlation.
# 4. Graph saved as 'tastytrade_correlation_heatmap.png' and also displayed.
################################################################################

TT_PRICE_HISTORY_URL = "https://api.tastytrade.com/instruments/equities/{symbol}/price-history"


def get_tastytrade_session() -> Optional[Session]:
    user = os.environ.get("TASTY_USER")
    pwd = os.environ.get("TASTY_PASS")
    if not user or not pwd:
        print("[WARN] TASTY_USER / TASTY_PASS not set in environment.")
        return None
    try:
        return Session(user, pwd)
    except Exception as e:
        print(f"[WARN] Could not establish TastyTrade session: {e}")
        return None


def fetch_tastytrade_history(session: Session, symbol: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    """Attempt to fetch daily close prices for symbol using raw REST call.
    Returns pandas Series indexed by date or None on failure."""
    try:
        # The Session object (from tastytrade lib) exposes an underlying requests.Session at .session
        # We attempt the assumed endpoint. If the tastytrade SDK changes, this may need adjustment.
        s = session.session  # type: ignore[attr-defined]
        url = TT_PRICE_HISTORY_URL.format(symbol=symbol.upper())
        params = {
            "start-date": start.strftime('%Y-%m-%d'),
            "end-date": end.strftime('%Y-%m-%d'),
            "interval": "1d"
        }
        resp = s.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  [TT FAIL] {symbol}: HTTP {resp.status_code}")
            return None
        data = resp.json()
        # Heuristic parse: look for key containing 'items' or 'priceHistory'
        items = None
        if isinstance(data, dict):
            for k in ("items", "data", "priceHistory", "prices"):
                if k in data and isinstance(data[k], list):
                    items = data[k]
                    break
            if items is None and 'result' in data and isinstance(data['result'], list):
                items = data['result']
        if not items:
            print(f"  [TT FAIL] {symbol}: Unexpected JSON schema")
            return None
        rows = []
        for row in items:
            # Common possible keys; adapt gracefully
            date_key = row.get('date') or row.get('time') or row.get('timestamp')
            close = (row.get('close') or row.get('close-price') or row.get('closePrice')
                     or row.get('last') or row.get('mark'))
            if not date_key or close is None:
                continue
            # Normalize date
            try:
                # Accept either date (YYYY-MM-DD) or timestamp
                if isinstance(date_key, (int, float)):
                    dt = datetime.utcfromtimestamp(int(date_key) / (1000 if int(date_key) > 10**12 else 1))
                else:
                    # Strip time if present
                    dt = datetime.fromisoformat(str(date_key).replace('Z', '').split('T')[0])
                rows.append((dt.date(), float(close)))
            except Exception:
                continue
        if not rows:
            print(f"  [TT FAIL] {symbol}: No parsable rows")
            return None
        # Deduplicate by date (keep last)
        dedup = {}
        for d, c in rows:
            dedup[d] = c
        series = pd.Series(dedup).sort_index()
        series.name = symbol.upper()
        return series
    except Exception as e:
        print(f"  [TT EXC] {symbol}: {e}")
        return None


def fetch_history(symbol: str, start: datetime, end: datetime, session: Optional[Session]) -> Optional[pd.Series]:
    if session:
        series = fetch_tastytrade_history(session, symbol, start, end)
        if series is not None:
            return series
    # Fallback yfinance
    try:
        print(f"  [YF] Fallback for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'), end=(end + timedelta(days=1)).strftime('%Y-%m-%d'), interval='1d')
        if df.empty:
            return None
        s = df['Close']
        s.index = pd.to_datetime(s.index).date
        s.name = symbol.upper()
        return s
    except Exception as e:
        print(f"  [YF FAIL] {symbol}: {e}")
        return None


def build_price_dataframe(symbols: List[str], start: datetime, end: datetime, session: Optional[Session]) -> pd.DataFrame:
    series_list = []
    for sym in symbols:
        print(f"Fetching {sym} ...")
        s = fetch_history(sym, start, end, session)
        if s is None or s.empty:
            print(f"  [SKIP] {sym} - no data")
            continue
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    return df


def compute_return_correlation(price_df: pd.DataFrame) -> pd.DataFrame:
    returns = price_df.pct_change().dropna(how='all')
    corr = returns.corr()
    return corr


def plot_correlation_heatmap(corr: pd.DataFrame, output_file: str) -> str:
    plt.figure(figsize=(1.2 * len(corr.columns) + 2, 1.0 * len(corr.columns) + 2))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True,
                cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='gray')
    plt.title('Daily Return Correlation')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Saved heatmap to {output_file}")
    try:
        plt.show()
    except Exception:
        pass
    return output_file


def main():
    print("=== TastyTrade Correlation Calculator ===")
    print("Enter comma-separated tickers (e.g. AAPL,MSFT,NVDA):")
    symbols_raw = input('Tickers: ').strip()
    if not symbols_raw:
        print('No tickers provided. Exiting.')
        return
    symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    if len(symbols) < 2:
        print('Need at least 2 symbols for correlation.')
        return

    default_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_str = input(f'Start date [YYYY-MM-DD] (default {default_start}): ').strip() or default_start
    end_str = input('End date [YYYY-MM-DD] (default today): ').strip() or datetime.today().strftime('%Y-%m-%d')

    try:
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
    except ValueError:
        print('Invalid date format. Use YYYY-MM-DD.')
        return

    if start_date >= end_date:
        print('Start date must be before end date.')
        return

    session = get_tastytrade_session()
    if session:
        print('[INFO] TastyTrade session established.')
    else:
        print('[INFO] Proceeding without TastyTrade session (will use yfinance fallback).')

    price_df = build_price_dataframe(symbols, start_date, end_date, session)
    if price_df.empty:
        print('No price data collected. Exiting.')
        return

    # Drop columns with insufficient data (e.g., all NaN)
    valid_price_df = price_df.dropna(axis=1, how='all')
    if valid_price_df.shape[1] < 2:
        print('Not enough valid symbols with data to compute correlation.')
        return

    corr = compute_return_correlation(valid_price_df)
    print('\nCorrelation Matrix (daily returns):')
    print(corr.round(3))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # outfile = f'tastytrade_correlation_heatmap_{ts}.png'
    # plot_correlation_heatmap(corr, outfile)

    print('\nDone.')


if __name__ == '__main__':
    main()
