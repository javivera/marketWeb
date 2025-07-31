#!/usr/bin/env python3
"""
Monte Carlo Portfolio Simulation Script
========================================
Simulates portfolio returns using Monte Carlo methods based on historical data.
Configuration is read from .env file.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import base64
import io
import time
import threading
from scipy import stats
import json
import shutil
import subprocess

warnings.filterwarnings('ignore')

# Simple .env file reader (no external dependencies)
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    env_vars = {}
    if not os.path.exists(env_path):
        print(f"⚠️  Warning: {env_path} file not found. Using default values.")
        return env_vars
    
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                env_vars[key.strip()] = value
    
    return env_vars

def parse_portfolio_holdings(holdings_string):
    """Parse portfolio holdings from string format"""
    holdings = {}
    if not holdings_string:
        return holdings
    
    try:
        for holding in holdings_string.split(','):
            if ':' in holding:
                symbol, shares = holding.strip().split(':')
                holdings[symbol.strip().upper()] = int(shares.strip())
    except Exception as e:
        print(f"⚠️  Error parsing portfolio holdings: {e}")
        print(f"   Expected format: SYMBOL:SHARES,SYMBOL:SHARES")
        print(f"   Example: AAPL:100,MSFT:50,GOOGL:25")
    
    return holdings

def get_stock_data(symbols, period="1y"):
    """Fetch stock data for multiple symbols"""
    print(f"📊 Fetching {period} of data for portfolio stocks...")
    
    stock_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            print(f"  • Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty or len(data) < 50:
                print(f"    ❌ {symbol}: Insufficient data")
                failed_symbols.append(symbol)
            else:
                stock_data[symbol] = data
                print(f"    ✅ {symbol}: {len(data)} trading days")
                
        except Exception as e:
            print(f"    ❌ {symbol}: Failed to download - {str(e)}")
            failed_symbols.append(symbol)
            continue
    
    if not stock_data:
        raise ValueError("No valid stock data retrieved. Please check your symbols.")
    
    return stock_data, failed_symbols

def calculate_daily_returns(stock_data):
    """Calculate daily returns for all stocks"""
    returns_data = {}
    
    for symbol, data in stock_data.items():
        if 'Close' in data.columns:
            prices = data['Close']
            daily_returns = prices.pct_change().dropna()
            returns_data[symbol] = daily_returns
    
    return pd.DataFrame(returns_data)

def run_monte_carlo_simulation(portfolio_holdings, stock_returns, num_simulations=1000, time_horizon=252):
    """Run Monte Carlo simulation for portfolio"""
    print(f"🎲 Running Monte Carlo simulation...")
    print(f"   • Simulations: {num_simulations:,}")
    print(f"   • Time horizon: {time_horizon} days")
    
    # Get current prices for portfolio valuation
    portfolio_symbols = list(portfolio_holdings.keys())
    current_prices = {}
    
    for symbol in portfolio_symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.history(period="5d")
            if not info.empty:
                current_prices[symbol] = info['Close'].iloc[-1]
            else:
                print(f"⚠️  Could not get current price for {symbol}")
                current_prices[symbol] = 100  # Default fallback
        except:
            print(f"⚠️  Could not get current price for {symbol}")
            current_prices[symbol] = 100  # Default fallback
    
    # Calculate current portfolio value
    current_portfolio_value = sum(
        portfolio_holdings[symbol] * current_prices[symbol] 
        for symbol in portfolio_symbols
    )
    
    # Calculate portfolio weights
    position_values = {
        symbol: portfolio_holdings[symbol] * current_prices[symbol]
        for symbol in portfolio_symbols
    }
    
    portfolio_weights = np.array([
        position_values[symbol] / current_portfolio_value
        for symbol in portfolio_symbols
    ])
    
    # Get returns for portfolio stocks
    portfolio_returns = stock_returns[portfolio_symbols].dropna()
    
    if portfolio_returns.empty:
        raise ValueError("No return data available for portfolio stocks")
    
    # Calculate portfolio statistics
    mean_returns = portfolio_returns.mean()
    cov_matrix = portfolio_returns.cov()
    
    # Run Monte Carlo simulation
    simulation_results = []
    
    for i in range(num_simulations):
        # Generate random returns based on multivariate normal distribution
        random_returns = np.random.multivariate_normal(
            mean_returns * time_horizon,  # Annualized returns
            cov_matrix * time_horizon,    # Annualized covariance
            1
        )[0]
        
        # Calculate portfolio return
        portfolio_return = np.dot(portfolio_weights, random_returns)
        simulation_results.append(portfolio_return * 100)  # Convert to percentage
    
    return np.array(simulation_results), current_portfolio_value, position_values, current_prices

def calculate_historical_performance(portfolio_holdings, stock_data, spy_data):
    """Calculate historical performance vs SPY by year"""
    print("📈 Calculating historical performance...")
    
    portfolio_symbols = list(portfolio_holdings.keys())
    current_prices = {}
    
    # Get current prices
    for symbol in portfolio_symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.history(period="5d")
            if not info.empty:
                current_prices[symbol] = info['Close'].iloc[-1]
            else:
                current_prices[symbol] = 100
        except:
            current_prices[symbol] = 100
    
    # Calculate portfolio weights
    current_portfolio_value = sum(
        portfolio_holdings[symbol] * current_prices[symbol] 
        for symbol in portfolio_symbols
    )
    
    portfolio_weights = {
        symbol: (portfolio_holdings[symbol] * current_prices[symbol]) / current_portfolio_value
        for symbol in portfolio_symbols
    }
    
    # Calculate historical performance by year
    years = [2020, 2021, 2022, 2023, 2024]
    performance_data = []
    
    for year in years:
        try:
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            
            # Calculate portfolio performance
            portfolio_return = 0
            valid_stocks = 0
            
            for symbol in portfolio_symbols:
                if symbol in stock_data:
                    stock_year_data = stock_data[symbol].loc[year_start:year_end]
                    if len(stock_year_data) > 0:
                        start_price = stock_year_data['Close'].iloc[0]
                        end_price = stock_year_data['Close'].iloc[-1]
                        stock_return = ((end_price - start_price) / start_price) * 100
                        portfolio_return += portfolio_weights[symbol] * stock_return
                        valid_stocks += 1
            
            # Calculate SPY performance
            spy_return = 0
            if 'SPY' in spy_data:
                spy_year_data = spy_data['SPY'].loc[year_start:year_end]
                if len(spy_year_data) > 0:
                    spy_start = spy_year_data['Close'].iloc[0]
                    spy_end = spy_year_data['Close'].iloc[-1]
                    spy_return = ((spy_end - spy_start) / spy_start) * 100
            
            performance_data.append({
                "period": str(year),
                "portfolio_return": portfolio_return,
                "spy_return": spy_return,
                "portfolio_value": current_portfolio_value * (1 + portfolio_return/100),
                "spy_equivalent": current_portfolio_value * (1 + spy_return/100)
            })
            
        except Exception as e:
            print(f"⚠️  Could not calculate performance for {year}: {e}")
    
    # Add YTD performance
    try:
        ytd_start = "2025-01-01"
        ytd_end = datetime.now().strftime("%Y-%m-%d")
        
        portfolio_return = 0
        for symbol in portfolio_symbols:
            if symbol in stock_data:
                stock_ytd_data = stock_data[symbol].loc[ytd_start:ytd_end]
                if len(stock_ytd_data) > 0:
                    start_price = stock_ytd_data['Close'].iloc[0]
                    end_price = stock_ytd_data['Close'].iloc[-1]
                    stock_return = ((end_price - start_price) / start_price) * 100
                    portfolio_return += portfolio_weights[symbol] * stock_return
        
        spy_return = 0
        if 'SPY' in spy_data:
            spy_ytd_data = spy_data['SPY'].loc[ytd_start:ytd_end]
            if len(spy_ytd_data) > 0:
                spy_start = spy_ytd_data['Close'].iloc[0]
                spy_end = spy_ytd_data['Close'].iloc[-1]
                spy_return = ((spy_end - spy_start) / spy_start) * 100
        
        performance_data.append({
            "period": "2025 YTD",
            "portfolio_return": portfolio_return,
            "spy_return": spy_return,
            "portfolio_value": current_portfolio_value * (1 + portfolio_return/100),
            "spy_equivalent": current_portfolio_value * (1 + spy_return/100)
        })
        
    except Exception as e:
        print(f"⚠️  Could not calculate YTD performance: {e}")
    
    return performance_data

def save_chart_to_file(fig, filename):
    """Save matplotlib figure to PNG file"""
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    filepath = assets_dir / f"{filename}.png"
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight', 
                facecolor='none', transparent=True)
    plt.close(fig)
    return f"assets/{filename}.png"

def create_simulation_charts(simulation_results, current_portfolio_value):
    """Create Monte Carlo simulation charts"""
    plt.style.use('dark_background')
    
    charts = []
    
    # Chart 1: Histogram of returns
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.hist(simulation_results, bins=50, alpha=0.7, color='skyblue', edgecolor='white', density=True)
    ax.axvline(np.mean(simulation_results), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(simulation_results):.2f}%')
    ax.axvline(np.percentile(simulation_results, 95), color='green', linestyle='--', linewidth=2,
               label=f'95th Percentile: {np.percentile(simulation_results, 95):.2f}%')
    ax.axvline(np.percentile(simulation_results, 5), color='orange', linestyle='--', linewidth=2,
               label=f'5th Percentile: {np.percentile(simulation_results, 5):.2f}%')
    
    ax.set_xlabel('Portfolio Return (%)', color='white', fontsize=12)
    ax.set_ylabel('Probability Density', color='white', fontsize=12)
    ax.set_title('Monte Carlo Simulation: Portfolio Return Distribution', color='white', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    charts.append({
        "title": "Portfolio Return Distribution",
        "image": save_chart_to_file(fig, "monte_carlo_return_distribution")
    })
    
    # Chart 2: Risk vs Return scatter
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create risk-return buckets
    returns_sorted = np.sort(simulation_results)
    n_buckets = 20
    bucket_size = len(returns_sorted) // n_buckets
    
    bucket_means = []
    bucket_stds = []
    
    for i in range(n_buckets):
        start_idx = i * bucket_size
        end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else len(returns_sorted)
        bucket_data = returns_sorted[start_idx:end_idx]
        bucket_means.append(np.mean(bucket_data))
        bucket_stds.append(np.std(bucket_data))
    
    ax.scatter(bucket_stds, bucket_means, alpha=0.6, c=bucket_means, cmap='RdYlGn', s=100)
    ax.set_xlabel('Risk (Standard Deviation %)', color='white', fontsize=12)
    ax.set_ylabel('Expected Return (%)', color='white', fontsize=12)
    ax.set_title('Risk vs Return Analysis', color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0])
    cbar.set_label('Return (%)', color='white')
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    charts.append({
        "title": "Risk vs Return Analysis",
        "image": save_chart_to_file(fig, "monte_carlo_risk_return")
    })
    
    # Chart 3: Cumulative probability
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_returns = np.sort(simulation_results)
    probabilities = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    ax.plot(sorted_returns, probabilities * 100, linewidth=2, color='cyan')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax.axhline(50, color='yellow', linestyle='--', alpha=0.7, label='50% Probability')
    
    ax.set_xlabel('Portfolio Return (%)', color='white', fontsize=12)
    ax.set_ylabel('Cumulative Probability (%)', color='white', fontsize=12)
    ax.set_title('Cumulative Probability Distribution', color='white', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    charts.append({
        "title": "Cumulative Probability Distribution",
        "image": save_chart_to_file(fig, "monte_carlo_cumulative_probability")
    })
    
    return charts

def generate_assets(portfolio_holdings, simulation_results, current_portfolio_value, 
                   position_values, current_prices, historical_performance, 
                   num_simulations, time_horizon, initial_cash):
    """Generate JSON data and chart assets"""
    print("📄 Generating assets...")
    
    # Ensure assets directory exists
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Generate charts
    charts = create_simulation_charts(simulation_results, current_portfolio_value)
    
    # Calculate simulation statistics
    mean_return = np.mean(simulation_results)
    median_return = np.median(simulation_results)
    std_return = np.std(simulation_results)
    var_5 = np.percentile(simulation_results, 5)
    var_1 = np.percentile(simulation_results, 1)
    max_drawdown = var_1  # Simplified
    
    positive_returns = simulation_results[simulation_results > 0]
    positive_probability = (len(positive_returns) / len(simulation_results)) * 100
    prob_above_10 = (len(simulation_results[simulation_results > 10]) / len(simulation_results)) * 100
    prob_below_neg10 = (len(simulation_results[simulation_results < -10]) / len(simulation_results)) * 100
    
    best_case = np.max(simulation_results)
    worst_case = np.min(simulation_results)
    
    # Prepare portfolio summary
    portfolio_summary = {
        "holdings": [
            {
                "symbol": symbol,
                "shares": shares,
                "current_price": current_prices[symbol],
                "position_value": position_values[symbol],
                "weight": (position_values[symbol] / current_portfolio_value) * 100
            }
            for symbol, shares in portfolio_holdings.items()
        ],
        "total_value": current_portfolio_value,
        "initial_cash": initial_cash
    }
    
    # Create comprehensive data object
    data = {
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_simulations": num_simulations,
        "simulation_days": time_horizon,
        "current_portfolio_value": current_portfolio_value,
        "portfolio_summary": portfolio_summary,
        "simulation_results": {
            "mean_return": mean_return,
            "median_return": median_return,
            "std_return": std_return,
            "var_5": var_5,
            "var_1": var_1,
            "max_drawdown": max_drawdown,
            "positive_probability": positive_probability,
            "prob_above_10": prob_above_10,
            "prob_below_neg10": prob_below_neg10,
            "best_case": best_case,
            "worst_case": worst_case
        },
        "charts": charts,
        "historical_performance": historical_performance
    }
    
    # Save to JSON file
    json_file = assets_dir / "monte_carlo_data.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"📄 Assets saved to: {assets_dir}")
    return str(json_file)

def copy_template_to_output():
    """Copy the HTML template to the final output location"""
    template_path = Path("templates/monte_carlo_portfolio.html")
    output_path = Path("montecarlo.html")
    
    if template_path.exists():
        shutil.copy2(template_path, output_path)
        print(f"📄 HTML template copied to: {output_path}")
        return str(output_path)
    else:
        print(f"⚠️  Template not found: {template_path}")
        return None

def run_git_commands():
    """Handle git operations (commit and push)"""
    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
        
        # Create commit
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Auto-update: Monte Carlo analysis - {timestamp}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True)
        print("✅ Git commit created successfully")
        
        # Push changes
        subprocess.run(['git', 'push'], check=True, capture_output=True)
        print("🚀 Changes pushed to git repository")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Git operation failed: {e}")
    except Exception as e:
        print(f"⚠️  Git error: {e}")

def scheduled_analysis():
    """Run the analysis periodically"""
    while True:
        try:
            print("\\n" + "="*80)
            print("🔄 SCHEDULED MONTE CARLO ANALYSIS STARTING")
            print("="*80)
            main()
            
            # Get schedule interval from env
            env_vars = load_env_file()
            interval_hours = int(env_vars.get('SCHEDULE_INTERVAL_HOURS', 4))
            
            print(f"\\n💤 Sleeping for {interval_hours} hours until next analysis...")
            time.sleep(interval_hours * 3600)  # Convert hours to seconds
            
        except KeyboardInterrupt:
            print("\\n🛑 Scheduled analysis stopped by user")
            break
        except Exception as e:
            print(f"\\n❌ Error in scheduled analysis: {e}")
            print("🔄 Retrying in 1 hour...")
            time.sleep(3600)

def main():
    """Main analysis function"""
    # Load configuration
    env_vars = load_env_file()
    
    # Configuration with defaults
    data_period = env_vars.get('DATA_PERIOD', '3y')
    portfolio_holdings_str = env_vars.get('PORTFOLIO_HOLDINGS', 'AAPL:100,MSFT:50,GOOGL:25')
    num_simulations = int(env_vars.get('MONTE_CARLO_SIMULATIONS', 1000))
    time_horizon = int(env_vars.get('SIMULATION_DAYS', 252))
    initial_cash = float(env_vars.get('INITIAL_CASH', 0))
    auto_git_push = env_vars.get('AUTO_GIT_PUSH', 'false').lower() == 'true'
    
    # Parse portfolio holdings
    portfolio_holdings = parse_portfolio_holdings(portfolio_holdings_str)
    
    if not portfolio_holdings:
        print("❌ No valid portfolio holdings found. Please check your PORTFOLIO_HOLDINGS in .env file.")
        print("   Expected format: SYMBOL:SHARES,SYMBOL:SHARES")
        print("   Example: AAPL:100,MSFT:50,GOOGL:25")
        return
    
    print("📋 Configuration loaded from .env:")
    print(f"   • Data Period: {data_period}")
    print(f"   • Simulations: {num_simulations:,}")
    print(f"   • Time Horizon: {time_horizon} days")
    print(f"   • Portfolio Holdings: {dict(portfolio_holdings)}")
    print(f"   • Initial Cash: ${initial_cash:,.2f}")
    
    print("="*80)
    print("STARTING MONTE CARLO PORTFOLIO SIMULATION")
    print(f"Portfolio: {len(portfolio_holdings)} stocks | Simulations: {num_simulations:,}")
    print("="*80)
    
    try:
        # Get all symbols including SPY for comparison
        all_symbols = list(portfolio_holdings.keys()) + ['SPY']
        
        # Fetch stock data
        stock_data, failed_symbols = get_stock_data(all_symbols, period=data_period)
        
        # Calculate daily returns
        daily_returns = calculate_daily_returns(stock_data)
        
        # Run Monte Carlo simulation
        simulation_results, current_portfolio_value, position_values, current_prices = run_monte_carlo_simulation(
            portfolio_holdings, daily_returns, num_simulations, time_horizon
        )
        
        # Calculate historical performance
        spy_data = {'SPY': stock_data.get('SPY')} if 'SPY' in stock_data else {}
        historical_performance = calculate_historical_performance(
            portfolio_holdings, stock_data, spy_data
        )
        
        # Generate assets
        json_file = generate_assets(
            portfolio_holdings, simulation_results, current_portfolio_value,
            position_values, current_prices, historical_performance,
            num_simulations, time_horizon, initial_cash
        )
        
        # Copy template to output location
        html_file = copy_template_to_output()
        
        if html_file:
            print(f"📄 HTML report saved as: {os.path.abspath(html_file)}")
            print("🌐 Open this file in your web browser to view the complete analysis")
        
        print("✅ Monte Carlo simulation complete!")
        
        # Git operations
        if auto_git_push:
            print("📦 Auto-committing to git...")
            run_git_commands()
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        print("📅 Starting scheduled Monte Carlo analysis...")
        print("   Press Ctrl+C to stop")
        
        # Start in a separate thread to allow for graceful shutdown
        analysis_thread = threading.Thread(target=scheduled_analysis, daemon=True)
        analysis_thread.start()
        
        try:
            analysis_thread.join()
        except KeyboardInterrupt:
            print("\\n🛑 Stopping scheduled analysis...")
    else:
        main()
