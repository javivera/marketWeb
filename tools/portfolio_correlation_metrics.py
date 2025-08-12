import os
import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from tastytrade import Session
import yfinance as yf

load_dotenv()

TT_PRICE_HISTORY_URL = "https://api.tastytrade.com/instruments/equities/{symbol}/price-history"

# ---------------------------- Data Fetching ---------------------------- #

def get_tastytrade_session() -> Optional[Session]:
    user = os.environ.get("TASTY_USER")
    pwd = os.environ.get("TASTY_PASS")
    if not user or not pwd:
        print("[WARN] TASTY_USER / TASTY_PASS not set.")
        return None
    try:
        return Session(user, pwd)
    except Exception as e:
        print(f"[WARN] Could not establish TastyTrade session: {e}")
        return None


def fetch_tastytrade_history(session: Session, symbol: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    try:
        s = session.session  # requests.Session (assumed)
        url = TT_PRICE_HISTORY_URL.format(symbol=symbol.upper())
        params = {
            "start-date": start.strftime('%Y-%m-%d'),
            "end-date": end.strftime('%Y-%m-%d'),
            "interval": "1d"
        }
        resp = s.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        items = None
        if isinstance(data, dict):
            for k in ("items", "data", "priceHistory", "prices"):
                if k in data and isinstance(data[k], list):
                    items = data[k]
                    break
            if items is None and 'result' in data and isinstance(data['result'], list):
                items = data['result']
        if not items:
            return None
        out = {}
        for row in items:
            date_key = row.get('date') or row.get('time') or row.get('timestamp')
            close = (row.get('close') or row.get('close-price') or row.get('closePrice')
                     or row.get('last') or row.get('mark'))
            if not date_key or close is None:
                continue
            try:
                if isinstance(date_key, (int, float)):
                    dt = datetime.utcfromtimestamp(int(date_key) / (1000 if int(date_key) > 10**12 else 1))
                else:
                    dt = datetime.fromisoformat(str(date_key).replace('Z', '').split('T')[0])
                out[dt.date()] = float(close)
            except Exception:
                continue
        if not out:
            return None
        ser = pd.Series(out).sort_index()
        ser.name = symbol.upper()
        return ser
    except Exception:
        return None


def fetch_history(symbol: str, start: datetime, end: datetime, session: Optional[Session]) -> Optional[pd.Series]:
    if session:
        ser = fetch_tastytrade_history(session, symbol, start, end)
        if ser is not None:
            return ser
    # fallback
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'), end=(end + timedelta(days=1)).strftime('%Y-%m-%d'), interval='1d')
        if df.empty:
            return None
        ser = df['Close']
        ser.index = pd.to_datetime(ser.index).date
        ser.name = symbol.upper()
        return ser
    except Exception:
        return None


def build_price_dataframe(symbols: List[str], start: datetime, end: datetime, session: Optional[Session]) -> pd.DataFrame:
    series = []
    for sym in symbols:
        print(f"Fetching {sym} ...")
        s = fetch_history(sym, start, end, session)
        if s is None or s.empty:
            print(f"  [SKIP] {sym}")
            continue
        series.append(s)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)

# ---------------------------- Metrics ---------------------------- #

def compute_core_matrices(price_df: pd.DataFrame):
    returns = price_df.pct_change().dropna(how='all')
    # remove columns fully NaN after pct_change
    returns = returns.dropna(axis=1, how='all')
    cov = returns.cov()
    corr = returns.corr()
    vols = returns.std()  # daily vol
    return returns, cov, corr, vols


def normalize_weights(symbols: List[str], raw_weights: List[float]) -> np.ndarray:
    if len(raw_weights) != len(symbols):
        raise ValueError("Weights length mismatch.")
    w = np.array(raw_weights, dtype=float)
    if (w < 0).any():
        raise ValueError("Negative weights not supported in this simple script.")
    if w.sum() == 0:
        raise ValueError("Sum of weights cannot be zero.")
    return w / w.sum()


def implied_average_correlation(w: np.ndarray, vols: np.ndarray, cov: pd.DataFrame) -> Optional[float]:
    # portfolio variance
    sigma_p2 = float(w @ cov.values @ w)
    S1 = float(np.sum(w * vols))
    S2 = float(np.sum((w ** 2) * (vols ** 2)))
    denom = (S1 ** 2 - S2)
    if denom <= 1e-16:
        return None
    rho_bar = (sigma_p2 - S2) / denom
    return rho_bar


def diversification_ratio(w: np.ndarray, vols: np.ndarray, cov: pd.DataFrame) -> float:
    sigma_p = math.sqrt(float(w @ cov.values @ w))
    S1 = float(np.sum(w * vols))
    return S1 / sigma_p if sigma_p > 0 else float('nan')


def eigen_concentration(cov: pd.DataFrame) -> Tuple[float, np.ndarray]:
    vals, _ = np.linalg.eigh(cov.values)  # eigh for symmetric
    vals = np.clip(vals, 0, None)
    total = vals.sum()
    if total <= 0:
        return float('nan'), vals
    # largest eigenvalue share
    return float(vals[-1] / total), vals[::-1]


def risk_contributions(w: np.ndarray, cov: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    Sigma_w = cov.values @ w
    sigma_p2 = float(w @ Sigma_w)
    if sigma_p2 <= 0:
        return np.zeros_like(w), np.zeros_like(w), 0.0
    rc = w * Sigma_w  # absolute contributions to variance
    pct = rc / sigma_p2
    return rc, pct, sigma_p2


def herfindahl_effective_number(pct_contrib: np.ndarray) -> float:
    H = float(np.sum(pct_contrib ** 2))
    return 1.0 / H if H > 0 else float('nan')


def simple_offdiag_average(corr: pd.DataFrame) -> float:
    n = corr.shape[0]
    if n < 2:
        return float('nan')
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = corr.values[mask]
    return float(np.nanmean(vals)) if vals.size else float('nan')

# ---------------------------- Reporting ---------------------------- #

def print_metrics(symbols: List[str], w: np.ndarray, vols: pd.Series, cov: pd.DataFrame, corr: pd.DataFrame):
    vols_arr = vols.values
    rho_bar = implied_average_correlation(w, vols_arr, cov)
    dr = diversification_ratio(w, vols_arr, cov)
    dr2 = dr ** 2 if np.isfinite(dr) else float('nan')
    offdiag_avg = simple_offdiag_average(corr)
    eig1_share, eigvals_desc = eigen_concentration(cov)
    rc_abs, rc_pct, sigma_p2 = risk_contributions(w, cov)
    eff_risk_contrib = herfindahl_effective_number(rc_pct)

    sigma_p = math.sqrt(sigma_p2) if sigma_p2 > 0 else float('nan')

    print("\n================ PORTFOLIO CORRELATION / DIVERSIFICATION METRICS ===============")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Weights: {', '.join(f'{x:.4f}' for x in w)} (sum=1)")
    print("-------------------------------------------------------------------------------")
    print(f"Portfolio volatility (daily): {sigma_p:.4%}")
    print(f"Weighted avg individual vol (Σ w_i σ_i): {np.sum(w * vols_arr):.4%}")
    print(f"Implied average correlation (vol-weighted): {rho_bar:.4f}" if rho_bar is not None else "Implied average correlation: N/A")
    print(f"Simple unweighted avg pairwise correlation: {offdiag_avg:.4f}")
    print(f"Diversification Ratio (DR): {dr:.4f}")
    print(f"Effective # uncorrelated bets (DR^2): {dr2:.4f}")
    print(f"First eigenvalue share of variance: {eig1_share:.4%}")
    print(f"Eigenvalues (descending, variance units): {', '.join(f'{v:.6f}' for v in eigvals_desc)}")
    print(f"Effective # risk contributors (1/∑p_i^2): {eff_risk_contrib:.4f}")
    print("-------------------------------------------------------------------------------")
    print("Risk Contributions (to variance):")
    print(f"{'Symbol':<10}{'Weight':>10}{'Vol%':>10}{'RC%Var':>12}{'RC%Tot':>12}")
    for sym, wi, voli, pcti in zip(symbols, w, vols_arr, rc_pct):
        print(f"{sym:<10}{wi:>10.4f}{voli:>10.4%}{pcti:>12.4%}{(pcti):>12.4%}")
    print("===============================================================================\n")

# ---------------------------- Main Flow ---------------------------- #

def parse_shares_input(symbols: List[str], shares_str: str) -> np.ndarray:
    """Parse user input for share counts.
    Accepts a single number (applied to all) or a comma-separated list matching symbols.
    Blank input defaults to 1 share each.
    """
    shares_str = shares_str.strip()
    if not shares_str:
        return np.ones(len(symbols), dtype=float)
    parts = [p.strip() for p in shares_str.split(',') if p.strip()]
    if len(parts) == 1:
        v = float(parts[0])
        if v < 0:
            raise ValueError("Share count cannot be negative.")
        return np.full(len(symbols), v, dtype=float)
    if len(parts) != len(symbols):
        raise ValueError("Share count list must match the number of symbols or be a single number.")
    vals = np.array([float(p) for p in parts], dtype=float)
    if (vals < 0).any():
        raise ValueError("Share count cannot be negative.")
    return vals


def weights_from_shares_array(symbols: List[str], shares: np.ndarray, price_df: pd.DataFrame) -> np.ndarray:
    """Compute value weights from share counts using the latest available price in price_df."""
    last_prices = price_df.ffill().bfill().iloc[-1].reindex(symbols).values
    values = shares * last_prices
    total = float(values.sum())
    if total <= 0:
        return np.array([1/len(symbols)] * len(symbols))
    return values / total


def main():
    print("=== Portfolio Correlation / Diversification Metrics ===")
    # Prefer .env configuration; fall back to prompts
    symbols_env = os.environ.get('STOCK_SYMBOLS', '').strip()
    if symbols_env:
        symbols_input = [t.strip().upper() for t in symbols_env.split(',') if t.strip()]
        print(f"[CFG] STOCK_SYMBOLS from .env: {', '.join(symbols_input)}")
    else:
        tickers_in = input("Enter comma-separated tickers (e.g. AAPL,MSFT,NVDA): ").strip()
        if not tickers_in:
            print("No tickers provided.")
            return
        symbols_input = [t.strip().upper() for t in tickers_in.split(',') if t.strip()]
    if len(symbols_input) < 2:
        print("Need at least 2 tickers.")
        return

    # Dates: allow optional START_DATE / END_DATE in .env
    default_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_env = os.environ.get('START_DATE', '').strip()
    end_env = os.environ.get('END_DATE', '').strip()
    if start_env and end_env:
        start_str = start_env
        end_str = end_env
        print(f"[CFG] START_DATE={start_str}, END_DATE={end_str} from .env")
    else:
        start_str = input(f"Start date [YYYY-MM-DD] (default {default_start}): ").strip() or default_start
        end_str = input("End date [YYYY-MM-DD] (default today): ").strip() or datetime.today().strftime('%Y-%m-%d')

    try:
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
    except ValueError:
        print("Bad date format.")
        return
    if start_date >= end_date:
        print("Start date must be before end date.")
        return

    # Shares: allow SHARE_COUNTS in .env, else prompt
    shares_env = os.environ.get('SHARE_COUNTS', '').strip()
    if shares_env:
        try:
            shares_requested = parse_shares_input(symbols_input, shares_env)
            print(f"[CFG] SHARE_COUNTS from .env applied.")
        except Exception as e:
            print(f"Invalid SHARE_COUNTS in .env ({e}). Using 1 share each.")
            shares_requested = np.ones(len(symbols_input), dtype=float)
    else:
        print(f"Symbol order: {', '.join(symbols_input)}")
        shares_in = input("Enter share counts per symbol (comma-separated or single number for all) [default 1 each]: ").strip()
        try:
            shares_requested = parse_shares_input(symbols_input, shares_in)
        except Exception as e:
            print(f"Invalid share input ({e}). Using 1 share each.")
            shares_requested = np.ones(len(symbols_input), dtype=float)

    session = get_tastytrade_session()
    if session:
        print("[INFO] TastyTrade session established.")
    else:
        print("[INFO] Proceeding without TastyTrade session (will use yfinance fallback).")

    # Build price data for input symbols
    price_df = build_price_dataframe(symbols_input, start_date, end_date, session)
    if price_df.empty:
        print("No price data.")
        return

    # Keep only columns with some data and finalize symbol list
    price_df = price_df.dropna(axis=1, how='all')
    symbols = list(price_df.columns)

    # Map entered share counts to the symbols that actually have data
    share_map = {sym: count for sym, count in zip(symbols_input, shares_requested)}
    shares_final = np.array([share_map.get(sym, 0.0) for sym in symbols], dtype=float)

    # Compute value weights from shares using latest prices
    w = weights_from_shares_array(symbols, shares_final, price_df)

    returns, cov, corr, vols = compute_core_matrices(price_df)
    if cov.shape[0] < 2:
        print("Not enough data for correlation metrics.")
        return

    print_metrics(symbols, w, vols, cov, corr)

    # Optional: show correlation heatmap
    show_heatmap = input("Show correlation heatmap? [y/N]: ").strip().lower() == 'y'
    if show_heatmap:
        plt.figure(figsize=(1.2 * len(symbols) + 2, 1.0 * len(symbols) + 2))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True,
                    cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='gray')
        plt.title('Daily Return Correlation')
        plt.tight_layout()
        plt.show()

    print("Done.")


if __name__ == '__main__':
    main()
