#!/usr/bin/env python3
"""
Portfolio Optimizer Script
==========================
Optimizes portfolio allocation using Monte Carlo simulation.
Configuration is read from .env file.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings
from pathlib import Path
import base64
import io
import time
import threading
from itertools import combinations
import random
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

def get_stock_data(symbols, period="1y"):
    """Fetch stock data for multiple symbols"""
    print(f"📊 Fetching {period} of data for {len(symbols)} stocks...")
    
    stock_data = {}
    current_prices = {}
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"  • [{i:2d}/{len(symbols)}] Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty or len(data) < 50:
                print(f"    ❌ {symbol}: Insufficient data")
                failed_symbols.append(symbol)
            else:
                stock_data[symbol] = data
                current_prices[symbol] = data['Close'].iloc[-1]
                print(f"    ✅ {symbol}: {len(data)} trading days, Price: ${current_prices[symbol]:.2f}")
                
        except Exception as e:
            print(f"    ❌ {symbol}: Failed to download - {str(e)}")
            failed_symbols.append(symbol)
            continue
    
    print(f"✅ Successfully loaded {len(stock_data)} stocks")
    if failed_symbols:
        print(f"❌ Failed to load {len(failed_symbols)} stocks: {', '.join(failed_symbols)}")
    
    return stock_data, current_prices, failed_symbols

def calculate_daily_returns(stock_data):
    """Calculate daily returns for all stocks"""
    returns_data = {}
    
    for symbol, data in stock_data.items():
        if 'Close' in data.columns:
            prices = data['Close']
            daily_returns = prices.pct_change().dropna()
            returns_data[symbol] = daily_returns
    
    return pd.DataFrame(returns_data)

def calculate_correlation_metric(symbols, correlation_matrix):
    """Calculate average correlation metric for a portfolio"""
    if len(symbols) < 2:
        return 0.0
    
    correlations = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            symbol1, symbol2 = symbols[i], symbols[j]
            if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                corr = correlation_matrix.loc[symbol1, symbol2]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
    
    return np.mean(correlations) if correlations else 0.0

def run_portfolio_optimization(stock_symbols, stock_data, returns_data, current_prices, 
                             num_combinations, budget_limit, shares_per_stock, 
                             num_simulations, time_horizon):
    """Run portfolio optimization with Monte Carlo simulation"""
    
    print(f"🎯 Starting portfolio optimization...")
    print(f"   • Stock universe: {len(stock_symbols)} stocks")
    print(f"   • Testing combinations: {num_combinations:,}")
    print(f"   • Budget limit: ${budget_limit:,}")
    print(f"   • Shares per stock: {shares_per_stock}")
    print(f"   • Monte Carlo simulations: {num_simulations}")
    print(f"   • Time horizon: {time_horizon} days")
    
    # Pre-filter stocks by budget (100 shares each)
    affordable_stocks = []
    for symbol in stock_symbols:
        if symbol in current_prices:
            cost_per_100_shares = current_prices[symbol] * shares_per_stock
            if cost_per_100_shares <= budget_limit:  # Single stock shouldn't exceed budget
                affordable_stocks.append(symbol)
    
    print(f"💰 {len(affordable_stocks)} stocks are affordable within budget")
    
    if len(affordable_stocks) < 5:
        print(f"⚠️  Warning: Only {len(affordable_stocks)} affordable stocks found. Need at least 5.")
        return []
    
    # Calculate correlation matrix for affordable stocks
    affordable_returns = returns_data[affordable_stocks].dropna()
    correlation_matrix = affordable_returns.corr()
    
    optimization_results = []
    tested_combinations = 0
    valid_portfolios = 0
    
    print("🔄 Testing random portfolio combinations...")
    
    # Generate random combinations
    for attempt in range(num_combinations * 3):  # Try more combinations to account for budget filtering
        if tested_combinations >= num_combinations:
            break
            
        # Random portfolio size (3-8 stocks)
        portfolio_size = random.randint(3, min(8, len(affordable_stocks)))
        portfolio_symbols = random.sample(affordable_stocks, portfolio_size)
        
        # Calculate portfolio cost
        portfolio_cost = sum(current_prices[symbol] * shares_per_stock for symbol in portfolio_symbols)
        
        # Skip if over budget
        if portfolio_cost > budget_limit:
            continue
            
        tested_combinations += 1
        
        if tested_combinations % 1000 == 0:
            print(f"   • Tested: {tested_combinations:,} | Valid: {valid_portfolios:,}")
        
        try:
            # Get returns for this portfolio
            portfolio_returns_data = affordable_returns[portfolio_symbols].dropna()
            
            if len(portfolio_returns_data) < 50:  # Need sufficient data
                continue
            
            # Calculate equal weights
            weights = np.array([1/len(portfolio_symbols)] * len(portfolio_symbols))
            
            # Portfolio statistics
            mean_returns = portfolio_returns_data.mean()
            cov_matrix = portfolio_returns_data.cov()
            
            # Run mini Monte Carlo simulation
            simulation_results = []
            for _ in range(num_simulations):
                random_returns = np.random.multivariate_normal(
                    mean_returns.values * time_horizon,
                    cov_matrix.values * time_horizon,
                    1
                )[0]
                portfolio_return = np.dot(weights, random_returns) * 100
                simulation_results.append(portfolio_return)
            
            # Calculate metrics
            expected_return = np.mean(simulation_results)
            volatility = np.std(simulation_results)
            sharpe_ratio = expected_return / volatility if volatility > 0 else 0
            var_5 = np.percentile(simulation_results, 5)
            correlation_metric = calculate_correlation_metric(portfolio_symbols, correlation_matrix)
            
            # Store result
            optimization_results.append({
                'symbols': portfolio_symbols,
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_5': var_5,
                'correlation': correlation_metric,
                'cost': portfolio_cost,
                'num_stocks': len(portfolio_symbols)
            })
            
            valid_portfolios += 1
            
        except Exception as e:
            continue
    
    print(f"✅ Optimization complete!")
    print(f"   • Total combinations tested: {tested_combinations:,}")
    print(f"   • Valid portfolios found: {valid_portfolios:,}")
    
    # Sort by Sharpe ratio (descending)
    optimization_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    return optimization_results

def save_chart_to_file(fig, filename):
    """Save matplotlib figure to PNG file"""
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    filepath = assets_dir / f"{filename}.png"
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight', 
                facecolor='none', transparent=True)
    plt.close(fig)
    return f"assets/{filename}.png"

def create_optimization_charts(optimization_results):
    """Create optimization analysis charts"""
    plt.style.use('dark_background')
    charts = []
    
    if not optimization_results or len(optimization_results) < 10:
        return charts
    
    # Extract data for top 100 portfolios
    top_results = optimization_results[:100]
    returns = [r['expected_return'] for r in top_results]
    volatilities = [r['volatility'] for r in top_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in top_results]
    correlations = [r['correlation'] for r in top_results]
    
    # Chart 1: Risk vs Return scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, cmap='RdYlGn', 
                        s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Volatility (Risk %)', color='white', fontsize=12)
    ax.set_ylabel('Expected Return (%)', color='white', fontsize=12)
    ax.set_title('Portfolio Optimization: Risk vs Return', color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio', color='white')
    cbar.ax.tick_params(colors='white')
    
    # Highlight top 5 portfolios
    for i in range(min(5, len(top_results))):
        ax.annotate(f'#{i+1}', (volatilities[i], returns[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   color='yellow', fontweight='bold')
    
    plt.tight_layout()
    charts.append({
        "title": "Risk vs Return Analysis",
        "image": save_chart_to_file(fig, "portfolio_risk_return")
    })
    
    # Chart 2: Sharpe Ratio distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.hist(sharpe_ratios, bins=30, alpha=0.7, color='skyblue', edgecolor='white')
    ax.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(sharpe_ratios):.3f}')
    ax.axvline(sharpe_ratios[0], color='green', linestyle='--', linewidth=2,
               label=f'Best: {sharpe_ratios[0]:.3f}')
    
    ax.set_xlabel('Sharpe Ratio', color='white', fontsize=12)
    ax.set_ylabel('Number of Portfolios', color='white', fontsize=12)
    ax.set_title('Sharpe Ratio Distribution', color='white', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    charts.append({
        "title": "Sharpe Ratio Distribution",
        "image": save_chart_to_file(fig, "portfolio_sharpe_distribution")
    })
    
    # Chart 3: Correlation vs Performance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(correlations, sharpe_ratios, c=returns, cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Average Correlation', color='white', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', color='white', fontsize=12)
    ax.set_title('Correlation vs Performance', color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Expected Return (%)', color='white')
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    charts.append({
        "title": "Correlation vs Performance",
        "image": save_chart_to_file(fig, "portfolio_correlation_performance")
    })
    
    return charts

def generate_assets(optimization_results, stock_symbols, num_combinations, 
                   budget_limit, shares_per_stock, num_simulations, time_horizon):
    """Generate JSON data and chart assets"""
    print("📄 Generating assets...")
    
    # Ensure assets directory exists
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Generate charts
    charts = create_optimization_charts(optimization_results)
    
    # Calculate summary statistics
    if optimization_results:
        avg_return = np.mean([r['expected_return'] for r in optimization_results])
        avg_volatility = np.mean([r['volatility'] for r in optimization_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in optimization_results])
        best_sharpe = optimization_results[0]['sharpe_ratio'] if optimization_results else 0
        worst_sharpe = optimization_results[-1]['sharpe_ratio'] if optimization_results else 0
    else:
        avg_return = avg_volatility = avg_sharpe = best_sharpe = worst_sharpe = 0
    
    # Prepare top portfolios (limit to 40 as requested)
    top_portfolios = []
    for i, result in enumerate(optimization_results[:40], 1):
        top_portfolios.append({
            "rank": i,
            "symbols": result['symbols'],
            "symbols_display": ', '.join(result['symbols']),
            "expected_return": result['expected_return'],
            "volatility": result['volatility'],
            "sharpe_ratio": result['sharpe_ratio'],
            "var_5": result['var_5'],
            "correlation": result['correlation'],
            "cost": result['cost'],
            "num_stocks": result['num_stocks']
        })
    
    # Create comprehensive data object
    data = {
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "total_stocks": len(stock_symbols),
            "combinations_tested": num_combinations,
            "budget_limit": budget_limit,
            "shares_per_stock": shares_per_stock,
            "monte_carlo_simulations": num_simulations,
            "simulation_days": time_horizon
        },
        "summary_stats": {
            "portfolios_found": len(optimization_results),
            "avg_return": avg_return,
            "avg_volatility": avg_volatility,
            "avg_sharpe": avg_sharpe,
            "best_sharpe": best_sharpe,
            "worst_sharpe": worst_sharpe
        },
        "top_portfolios": top_portfolios,
        "charts": charts
    }
    
    # Save to JSON file
    json_file = assets_dir / "portfolio_optimization_data.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"📄 Assets saved to: {assets_dir}")
    return str(json_file)

def copy_template_to_output():
    """Copy the HTML template to the final output location"""
    template_path = Path("templates/portfolio_optimization.html")
    output_path = Path("portfolio_optimization.html")
    
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
        commit_message = f"Auto-update: Portfolio optimization - {timestamp}"
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
            print("🔄 SCHEDULED PORTFOLIO OPTIMIZATION STARTING")
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
    """Main optimization function"""
    # Load configuration
    env_vars = load_env_file()
    
    # Configuration with defaults
    data_period = env_vars.get('DATA_PERIOD', '1y')
    stock_list_str = env_vars.get('STOCK_LIST', 'AAPL,MSFT,GOOGL,TSLA,AMZN,META,NVDA,NFLX,AMD,INTC')
    num_combinations = int(env_vars.get('RANDOM_COMBINATIONS', 15000))
    budget_limit = float(env_vars.get('BUDGET_LIMIT', 70000))
    shares_per_stock = int(env_vars.get('SHARES_PER_STOCK', 100))
    num_simulations = int(env_vars.get('MONTE_CARLO_SIMULATIONS', 500))
    time_horizon = int(env_vars.get('SIMULATION_DAYS', 252))
    auto_git_push = env_vars.get('AUTO_GIT_PUSH', 'false').lower() == 'true'
    
    # Parse stock list
    stock_symbols = [symbol.strip().upper() for symbol in stock_list_str.split(',') if symbol.strip()]
    
    if not stock_symbols:
        print("❌ No valid stock symbols found. Please check your STOCK_LIST in .env file.")
        return
    
    print("📋 Configuration loaded from .env:")
    print(f"   • Data Period: {data_period}")
    print(f"   • Stock Universe: {len(stock_symbols)} stocks")
    print(f"   • Random Combinations: {num_combinations:,}")
    print(f"   • Budget Limit: ${budget_limit:,.2f}")
    print(f"   • Shares per Stock: {shares_per_stock}")
    print(f"   • Monte Carlo Sims: {num_simulations:,}")
    print(f"   • Time Horizon: {time_horizon} days")
    
    print("="*80)
    print("STARTING PORTFOLIO OPTIMIZATION")
    print(f"Universe: {len(stock_symbols)} stocks | Budget: ${budget_limit:,.0f}")
    print("="*80)
    
    try:
        # Fetch stock data
        stock_data, current_prices, failed_symbols = get_stock_data(stock_symbols, period=data_period)
        
        if not stock_data:
            print("❌ No stock data available. Cannot proceed with optimization.")
            return
        
        # Calculate daily returns
        returns_data = calculate_daily_returns(stock_data)
        
        # Run portfolio optimization
        optimization_results = run_portfolio_optimization(
            stock_symbols, stock_data, returns_data, current_prices,
            num_combinations, budget_limit, shares_per_stock, 
            num_simulations, time_horizon
        )
        
        if not optimization_results:
            print("❌ No valid portfolios found. Try adjusting your parameters.")
            return
        
        # Generate assets
        json_file = generate_assets(
            optimization_results, stock_symbols, num_combinations,
            budget_limit, shares_per_stock, num_simulations, time_horizon
        )
        
        # Copy template to output location
        html_file = copy_template_to_output()
        
        if html_file:
            print(f"📄 HTML report saved as: {os.path.abspath(html_file)}")
            print("🌐 Open this file in your web browser to view the complete analysis")
        
        # Show top 5 results
        print("\\n🏆 TOP 5 OPTIMIZED PORTFOLIOS:")
        print("-" * 120)
        print(f"{'Rank':<4} {'Stocks':<35} {'Return%':<8} {'Risk%':<8} {'Sharpe':<8} {'VaR-5%':<8} {'Corr':<6} {'Cost':<10}")
        print("-" * 120)
        
        for i, result in enumerate(optimization_results[:5], 1):
            stocks_str = ','.join(result['symbols'])
            if len(stocks_str) > 32:
                stocks_str = stocks_str[:29] + "..."
            
            print(f"{i:<4} {stocks_str:<35} {result['expected_return']:>7.2f} "
                  f"{result['volatility']:>7.2f} {result['sharpe_ratio']:>7.3f} "
                  f"{result['var_5']:>7.2f} {result['correlation']:>5.3f} "
                  f"${result['cost']:>9,.0f}")
        
        print("✅ Portfolio optimization complete!")
        
        # Git operations
        if auto_git_push:
            print("📦 Auto-committing to git...")
            run_git_commands()
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        print("📅 Starting scheduled portfolio optimization...")
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
