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
from scipy import stats
import json

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
        pairs = holdings_string.split(',')
        for pair in pairs:
            if ':' in pair:
                ticker, shares = pair.strip().split(':')
                holdings[ticker.strip()] = int(shares.strip())
    except Exception as e:
        print(f"❌ Error parsing portfolio holdings: {e}")
        print(f"   Expected format: TICKER:SHARES,TICKER:SHARES")
        return {}
    
    return holdings

def get_stock_data(symbols, period="1y"):
    """Fetch historical stock data"""
    print(f"📊 Fetching {period} of data for {len(symbols)} stocks...")
    
    successful_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            print(f"  • Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"    ❌ {symbol}: No data available")
                failed_symbols.append(symbol)
                continue
                
            successful_data[symbol] = data['Close']
            print(f"    ✅ {symbol}: {len(data)} trading days")
            
        except Exception as e:
            print(f"    ❌ {symbol}: Error - {e}")
            failed_symbols.append(symbol)
    
    if not successful_data:
        print("❌ No data could be fetched for any symbols")
        return None, failed_symbols
    
    # Combine all stock data into a single DataFrame
    price_data = pd.DataFrame(successful_data)
    price_data = price_data.dropna()
    
    return price_data, failed_symbols

def calculate_daily_returns(price_data):
    """
    Calculate daily returns for each stock - much simpler and more intuitive
    Daily returns are the standard approach for Monte Carlo simulations
    """
    print(f"📈 Calculating daily returns...")
    returns_data = {}
    
    for symbol, prices in price_data.items():
        if len(prices) >= 2:
            # Calculate daily returns: (price_today - price_yesterday) / price_yesterday
            daily_returns = prices.pct_change().dropna().tolist()
            returns_data[symbol] = daily_returns
            print(f"    ✅ {symbol}: {len(daily_returns)} daily returns")
        else:
            print(f"⚠️  {symbol}: Insufficient data for daily returns")
    
    return returns_data

def monte_carlo_simulation(returns_data, portfolio_holdings, current_prices, 
                         num_simulations, simulation_days, initial_cash=0):
    """
    Run Monte Carlo simulation for portfolio returns using daily returns
    
    Parameters:
    - returns_data: Daily returns for each stock
    - portfolio_holdings: Dictionary of {symbol: shares}
    - current_prices: Current prices for each stock
    - num_simulations: Number of Monte Carlo runs
    - simulation_days: Number of days to simulate into the future
    - initial_cash: Additional cash in the portfolio
    """
    
    print(f"🎲 Running Monte Carlo simulation...")
    print(f"   • Simulations: {num_simulations:,}")
    print(f"   • Time horizon: {simulation_days} days ({simulation_days/252:.2f} years)")
    print(f"   • Portfolio stocks: {len(portfolio_holdings)}")
    print(f"   • Using daily returns for maximum flexibility")
    
    # Calculate current portfolio value and weights
    portfolio_value = initial_cash
    valid_holdings = {}
    stock_values = {}
    
    for symbol, shares in portfolio_holdings.items():
        if symbol in current_prices and symbol in returns_data:
            stock_value = shares * current_prices[symbol]
            portfolio_value += stock_value
            valid_holdings[symbol] = shares
            stock_values[symbol] = stock_value
            print(f"   • {symbol}: {shares} shares @ ${current_prices[symbol]:.2f} = ${stock_value:,.2f}")
        else:
            print(f"   ⚠️  {symbol}: No data available, excluding from simulation")
    
    print(f"   • Total initial portfolio value: ${portfolio_value:,.2f}")
    
    if not valid_holdings:
        print("❌ No valid holdings found for simulation")
        return None
    
    # Calculate portfolio weights
    portfolio_weights = {symbol: stock_values[symbol] / portfolio_value for symbol in valid_holdings}
    
    # Prepare returns statistics for simulation
    stocks = list(valid_holdings.keys())
    returns_matrix = np.array([returns_data[stock] for stock in stocks]).T
    
    # Calculate mean daily returns and covariance matrix (already daily, no conversion needed)
    mean_daily_returns = np.mean(returns_matrix, axis=0)
    daily_cov_matrix = np.cov(returns_matrix.T)
    
    # Add small amount to diagonal for numerical stability
    daily_cov_matrix += np.eye(len(daily_cov_matrix)) * 1e-8
    
    print(f"   • Mean daily returns: {[f'{r*100:.4f}%' for r in mean_daily_returns]}")
    print(f"   • Portfolio weights: {[f'{symbol}: {w:.1%}' for symbol, w in portfolio_weights.items()]}")
    print(f"   • Running {num_simulations:,} simulations...")
    
    # Run Monte Carlo simulations
    simulation_results = []
    
    for sim in range(num_simulations):
        # Track portfolio value over time
        portfolio_values = [portfolio_value]
        current_portfolio_value = portfolio_value
        
        for day in range(simulation_days):
            # Generate correlated random daily returns
            try:
                random_daily_returns = np.random.multivariate_normal(mean_daily_returns, daily_cov_matrix)
            except np.linalg.LinAlgError:
                # Fallback to independent returns if covariance matrix is singular
                random_daily_returns = np.random.normal(mean_daily_returns, np.sqrt(np.diag(daily_cov_matrix)))
            
            # Calculate portfolio return for this day
            portfolio_daily_return = 0
            for i, symbol in enumerate(stocks):
                weight = portfolio_weights[symbol]
                portfolio_daily_return += weight * random_daily_returns[i]
            
            # Apply daily return to portfolio
            current_portfolio_value *= (1 + portfolio_daily_return)
            portfolio_values.append(current_portfolio_value)
        
        # Calculate final return
        final_return = (current_portfolio_value - portfolio_value) / portfolio_value
        simulation_results.append({
            'final_value': current_portfolio_value,
            'total_return': final_return,
            'portfolio_values': portfolio_values
        })
        
        if (sim + 1) % 1000 == 0:
            print(f"   • Completed {sim + 1:,} simulations...")
    
    return {
        'initial_value': portfolio_value,
        'simulations': simulation_results,
        'portfolio_holdings': valid_holdings,
        'portfolio_weights': portfolio_weights,
        'stocks': stocks,
        'mean_daily_returns': mean_daily_returns,
        'daily_cov_matrix': daily_cov_matrix,
        'simulation_days': simulation_days
    }

def analyze_simulation_results(simulation_data, config=None):
    """Analyze Monte Carlo simulation results with proper annualization"""
    
    if not simulation_data:
        return None
    
    results = simulation_data['simulations']
    initial_value = simulation_data['initial_value']
    simulation_days = simulation_data.get('simulation_days', 252)
    
    # Extract results
    final_values = [r['final_value'] for r in results]
    total_returns = [r['total_return'] for r in results]
    
    # Calculate basic statistics for the simulation period
    mean_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    
    # Calculate annualized returns properly based on simulation period
    periods_per_year = 252 / simulation_days
    annualized_return = ((1 + mean_return) ** periods_per_year) - 1
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    
    # Sharpe ratio (get risk-free rate from config)
    risk_free_rate = float(config.get('RISK_FREE_RATE', 0.05)) if config else 0.05
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Calculate statistics
    analysis = {
        'initial_value': initial_value,
        'num_simulations': len(results),
        'simulation_days': simulation_days,
        'simulation_years': simulation_days / 252,
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'std_final_value': np.std(final_values),
        'mean_return': mean_return,
        'median_return': np.median(total_returns),
        'std_return': std_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'min_return': np.min(total_returns),
        'max_return': np.max(total_returns),
        'percentile_5': np.percentile(total_returns, 5),
        'percentile_25': np.percentile(total_returns, 25),
        'percentile_75': np.percentile(total_returns, 75),
        'percentile_95': np.percentile(total_returns, 95),
        'prob_loss': np.mean(np.array(total_returns) < 0),
        'prob_gain': np.mean(np.array(total_returns) > 0),
        'var_95': np.percentile(total_returns, 5),  # Value at Risk (95% confidence)
        'var_99': np.percentile(total_returns, 1),  # Value at Risk (99% confidence)
    }
    
    # Print enhanced summary
    print(f"\n📊 Monte Carlo Analysis Results ({simulation_days}-day / {simulation_days/252:.1f}-year simulation):")
    print(f"   • Expected return: {mean_return:.2%} ({annualized_return:.2%} annualized)")
    print(f"   • Volatility: {std_return:.2%} ({annualized_volatility:.2%} annualized)")
    print(f"   • Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"   • Probability of gain: {analysis['prob_gain']:.1%}")
    print(f"   • 95% VaR: {analysis['var_95']:.2%}")
    print(f"   • Return range: {analysis['min_return']:.2%} to {analysis['max_return']:.2%}")
    
    return analysis

def calculate_historical_performance(portfolio_holdings, current_prices):
    """
    Calculate historical portfolio performance for each year from 2020 to YTD
    Including SPY benchmark comparison
    """
    print(f"📊 Calculating historical portfolio performance...")
    
    # Define years to analyze
    years = [2020, 2021, 2022, 2023, 2024, 2025]  # 2025 will be YTD
    results = {}
    
    # Get portfolio symbols
    symbols = list(portfolio_holdings.keys())
    
    for year in years:
        print(f"   • Analyzing {year}...")
        
        if year == 2025:
            # Year to date - from Jan 1, 2025 to now
            start_date = f"{year}-01-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
            period_name = f"{year} YTD"
        else:
            # Full year
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            period_name = str(year)
        
        try:
            # Download SPY data for benchmark comparison
            spy_data = None
            try:
                spy_ticker = yf.Ticker("SPY")
                spy_hist = spy_ticker.history(start=start_date, end=end_date)
                if len(spy_hist) > 0:
                    spy_start = spy_hist['Close'].iloc[0]
                    spy_end = spy_hist['Close'].iloc[-1]
                    spy_return = (spy_end - spy_start) / spy_start
                    spy_data = {
                        'start_price': spy_start,
                        'end_price': spy_end,
                        'return': spy_return
                    }
                    print(f"     📊 SPY benchmark: {spy_return:.2%}")
            except Exception as e:
                print(f"     ⚠️  Could not fetch SPY data for {period_name}")
            
            # Download portfolio data for this period
            period_data = {}
            valid_symbols = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if len(hist) > 0:
                        period_data[symbol] = hist['Close']
                        valid_symbols.append(symbol)
                    else:
                        print(f"     ⚠️  {symbol}: No data available for {period_name}")
                        
                except Exception as e:
                    print(f"     ⚠️  {symbol}: Error fetching data for {period_name}")
                    continue
            
            if not valid_symbols:
                print(f"     ❌ No valid data for {period_name}")
                results[period_name] = None
                continue
            
            # Calculate portfolio performance
            portfolio_start_value = 0
            portfolio_end_value = 0
            stock_performances = {}
            
            for symbol in valid_symbols:
                if symbol in portfolio_holdings:
                    shares = portfolio_holdings[symbol]
                    prices = period_data[symbol]
                    
                    if len(prices) >= 2:
                        start_price = prices.iloc[0]
                        end_price = prices.iloc[-1]
                        
                        stock_start_value = shares * start_price
                        stock_end_value = shares * end_price
                        stock_return = (end_price - start_price) / start_price
                        
                        portfolio_start_value += stock_start_value
                        portfolio_end_value += stock_end_value
                        
                        stock_performances[symbol] = {
                            'shares': shares,
                            'start_price': start_price,
                            'end_price': end_price,
                            'return': stock_return,
                            'start_value': stock_start_value,
                            'end_value': stock_end_value
                        }
            
            if portfolio_start_value > 0:
                portfolio_return = (portfolio_end_value - portfolio_start_value) / portfolio_start_value
                
                # Calculate outperformance vs SPY
                outperformance = None
                if spy_data:
                    outperformance = portfolio_return - spy_data['return']
                
                results[period_name] = {
                    'start_value': portfolio_start_value,
                    'end_value': portfolio_end_value,
                    'return': portfolio_return,
                    'stock_performances': stock_performances,
                    'trading_days': len(period_data[valid_symbols[0]]) if valid_symbols else 0,
                    'spy_return': spy_data['return'] if spy_data else None,
                    'outperformance': outperformance
                }
                
                if outperformance is not None:
                    outperf_sign = "+" if outperformance >= 0 else ""
                    print(f"     ✅ {period_name}: {portfolio_return:.2%} return (${portfolio_start_value:,.0f} → ${portfolio_end_value:,.0f}) | SPY: {spy_data['return']:.2%} | Outperformance: {outperf_sign}{outperformance:.2%}")
                else:
                    print(f"     ✅ {period_name}: {portfolio_return:.2%} return (${portfolio_start_value:,.0f} → ${portfolio_end_value:,.0f})")
            else:
                results[period_name] = None
                print(f"     ❌ Could not calculate portfolio value for {period_name}")
                
        except Exception as e:
            print(f"     ❌ Error analyzing {period_name}: {str(e)}")
            results[period_name] = None
    
    return results

def generate_historical_performance_html(historical_performance):
    """Generate HTML section for historical performance with SPY comparison"""
    
    if not historical_performance:
        return ""
    
    # Filter out None results and sort by year
    valid_results = {k: v for k, v in historical_performance.items() if v is not None}
    
    if not valid_results:
        return "<h2>📈 Historical Portfolio Performance</h2><p>No historical data available.</p>"
    
    html = """
            <h2>📈 Historical Portfolio Performance vs SPY</h2>
            <p>How your current portfolio performed compared to the S&P 500 benchmark:</p>
            
            <div class="summary-grid">
    """
    
    # Sort by year (handle YTD specially)
    sorted_periods = sorted(valid_results.keys(), key=lambda x: (int(x.split()[0]), 0 if 'YTD' in x else 1))
    
    for period in sorted_periods:
        data = valid_results[period]
        return_pct = data['return']
        start_value = data['start_value']
        end_value = data['end_value']
        spy_return = data.get('spy_return')
        outperformance = data.get('outperformance')
        
        # Determine color based on performance
        color_class = 'positive' if return_pct > 0 else 'negative'
        emoji = '📈' if return_pct > 0 else '📉'
        
        # Outperformance indicators
        if outperformance is not None:
            if outperformance > 0:
                outperf_emoji = '🚀'
                outperf_class = 'positive'
                outperf_text = f'+{outperformance:.2%}'
            else:
                outperf_emoji = '📉'
                outperf_class = 'negative'
                outperf_text = f'{outperformance:.2%}'
        else:
            outperf_emoji = '❓'
            outperf_class = ''
            outperf_text = 'N/A'
        
        html += f"""
                <div class="summary-card">
                    <h3>{emoji} {period}</h3>
                    <p><strong>Portfolio:</strong> <span class="value {color_class}">{return_pct:.2%}</span></p>
        """
        
        if spy_return is not None:
            spy_color = 'positive' if spy_return > 0 else 'negative'
            html += f"""
                    <p><strong>SPY:</strong> <span class="value {spy_color}">{spy_return:.2%}</span></p>
                    <p><strong>Outperformance:</strong> <span class="value {outperf_class}">{outperf_text}</span> {outperf_emoji}</p>
            """
        
        html += f"""
                    <p><strong>Value:</strong> ${start_value:,.0f} → ${end_value:,.0f}</p>
                </div>
        """
    
    html += """
            </div>
            
            <h3>📊 Detailed Performance Comparison Table</h3>
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Portfolio Return</th>
                        <th>SPY Return</th>
                        <th>Outperformance</th>
                        <th>Portfolio Value</th>
                        <th>Gain/Loss</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for period in sorted_periods:
        data = valid_results[period]
        return_pct = data['return']
        start_value = data['start_value']
        end_value = data['end_value']
        gain_loss = end_value - start_value
        spy_return = data.get('spy_return')
        outperformance = data.get('outperformance')
        
        color_class = 'positive' if return_pct > 0 else 'negative'
        
        # SPY column
        spy_cell = f'<span class="value {"positive" if spy_return > 0 else "negative"}">{spy_return:.2%}</span>' if spy_return is not None else 'N/A'
        
        # Outperformance column
        if outperformance is not None:
            outperf_sign = '+' if outperformance >= 0 else ''
            outperf_class = 'positive' if outperformance >= 0 else 'negative'
            outperf_cell = f'<span class="value {outperf_class}">{outperf_sign}{outperformance:.2%}</span>'
        else:
            outperf_cell = 'N/A'
        
        html += f"""
                    <tr>
                        <td><strong>{period}</strong></td>
                        <td><span class="value {color_class}">{return_pct:.2%}</span></td>
                        <td>{spy_cell}</td>
                        <td>{outperf_cell}</td>
                        <td>${start_value:,.0f} → ${end_value:,.0f}</td>
                        <td><span class="value {color_class}">${gain_loss:+,.0f}</span></td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
            
            <div style="margin-top: 20px; padding: 15px; background-color: #f0f8ff; border-left: 4px solid #007acc; border-radius: 5px;">
                <h4>📝 Performance Notes:</h4>
                <ul>
                    <li><strong>Outperformance:</strong> Positive values mean your portfolio beat SPY, negative means it underperformed</li>
                    <li><strong>SPY Benchmark:</strong> Represents the S&P 500 index performance for the same periods</li>
                    <li><strong>🚀 = Outperformed SPY</strong> | <strong>📉 = Underperformed SPY</strong></li>
                </ul>
            </div>
    """
    
    return html

def create_simulation_charts(simulation_data, analysis):
    """Create visualization charts for Monte Carlo results"""
    
    if not simulation_data or not analysis:
        return None, None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of Final Returns
    returns = [r['total_return'] for r in simulation_data['simulations']]
    
    ax1.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(analysis['mean_return'], color='red', linestyle='--', 
                label=f'Mean: {analysis["mean_return"]:.2%}')
    ax1.axvline(analysis['median_return'], color='orange', linestyle='--', 
                label=f'Median: {analysis["median_return"]:.2%}')
    ax1.axvline(analysis['var_95'], color='darkred', linestyle='--', 
                label=f'VaR 95%: {analysis["var_95"]:.2%}')
    ax1.set_xlabel('Total Return')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Portfolio Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Portfolio Value Distribution
    final_values = [r['final_value'] for r in simulation_data['simulations']]
    
    ax2.hist(final_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(analysis['mean_final_value'], color='red', linestyle='--', 
                label=f'Mean: ${analysis["mean_final_value"]:,.0f}')
    ax2.axvline(analysis['initial_value'], color='black', linestyle='-', 
                label=f'Initial: ${analysis["initial_value"]:,.0f}')
    ax2.set_xlabel('Final Portfolio Value ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Final Portfolio Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sample Portfolio Paths (first 100 simulations)
    sample_paths = [r['portfolio_values'] for r in simulation_data['simulations'][:100]]
    days = range(len(sample_paths[0]))
    
    for path in sample_paths:
        ax3.plot(days, path, alpha=0.1, color='blue')
    
    # Plot mean path
    mean_path = np.mean([r['portfolio_values'] for r in simulation_data['simulations']], axis=0)
    ax3.plot(days, mean_path, color='red', linewidth=2, label='Mean Path')
    ax3.axhline(analysis['initial_value'], color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_title('Sample Portfolio Paths (100 simulations)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk Metrics
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(returns, p) for p in percentiles]
    
    ax4.bar(range(len(percentiles)), [v * 100 for v in percentile_values], 
            color=['red' if p <= 10 else 'orange' if p <= 25 else 'green' if p >= 75 else 'lightblue' 
                   for p in percentiles])
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Return (%)')
    ax4.set_title('Return Percentiles')
    ax4.set_xticks(range(len(percentiles)))
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_b64, analysis

def generate_html_report(simulation_data, analysis, chart_b64, config, historical_performance=None):
    """Generate HTML report for Monte Carlo simulation results"""
    
    if not analysis:
        return "<html><body><h1>No simulation results to display</h1></body></html>"
    
    # Portfolio composition
    portfolio_html = ""
    for symbol, shares in simulation_data['portfolio_holdings'].items():
        portfolio_html += f"<tr><td>{symbol}</td><td>{shares:,}</td></tr>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Monte Carlo Portfolio Simulation</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .summary-card {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .summary-card h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .summary-card .value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #3498db;
            }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }}
            .risk-metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .risk-card {{
                background-color: #fff;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #ddd;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎲 Monte Carlo Portfolio Simulation</h1>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>📊 Simulation Overview</h3>
                    <p><strong>Simulations:</strong> <span class="value">{analysis['num_simulations']:,}</span></p>
                    <p><strong>Time Horizon:</strong> <span class="value">{analysis['simulation_days']} days ({analysis['simulation_years']:.2f} years)</span></p>
                    <p><em>Using daily returns for maximum precision and flexibility. Choose any simulation period!</em></p>
                </div>
                
                <div class="summary-card">
                    <h3>💰 Portfolio Value</h3>
                    <p><strong>Initial:</strong> <span class="value">${analysis['initial_value']:,.0f}</span></p>
                    <p><strong>Mean Final:</strong> <span class="value">${analysis['mean_final_value']:,.0f}</span></p>
                    <p><strong>Median Final:</strong> <span class="value">${analysis['median_final_value']:,.0f}</span></p>
                </div>
                
                <div class="summary-card">
                    <h3>📈 Expected Returns</h3>
                    <p><strong>Mean Return:</strong> <span class="value {'positive' if analysis['mean_return'] > 0 else 'negative'}">{analysis['mean_return']:.2%}</span></p>
                    <p><strong>Annualized Return:</strong> <span class="value {'positive' if analysis['annualized_return'] > 0 else 'negative'}">{analysis['annualized_return']:.2%}</span></p>
                    <p><strong>Volatility (Annual):</strong> <span class="value">{analysis['annualized_volatility']:.2%}</span></p>
                    <p><strong>Sharpe Ratio:</strong> <span class="value {'positive' if analysis['sharpe_ratio'] > 0 else 'negative'}">{analysis['sharpe_ratio']:.2f}</span></p>
                </div>
                
                <div class="summary-card">
                    <h3>⚠️ Risk Metrics</h3>
                    <p><strong>VaR (95%):</strong> <span class="value negative">{analysis['var_95']:.2%}</span></p>
                    <p><strong>VaR (99%):</strong> <span class="value negative">{analysis['var_99']:.2%}</span></p>
                    <p><strong>Prob. of Gain:</strong> <span class="value">{analysis['prob_gain']:.1%}</span></p>
                    <p><strong>Simulation Period:</strong> <span class="value">{analysis['simulation_days']} days ({analysis['simulation_years']:.1f} years)</span></p>
                </div>
            </div>
            
            <h2>🏢 Portfolio Composition</h2>
            <table>
                <thead>
                    <tr>
                        <th>Stock Symbol</th>
                        <th>Shares</th>
                    </tr>
                </thead>
                <tbody>
                    {portfolio_html}
                </tbody>
            </table>
            
            <h2>📊 Detailed Statistics</h2>
            <div class="risk-metrics">
                <div class="risk-card">
                    <h4>Return Range</h4>
                    <p><strong>Min:</strong> <span class="negative">{analysis['min_return']:.2%}</span></p>
                    <p><strong>Max:</strong> <span class="positive">{analysis['max_return']:.2%}</span></p>
                </div>
                <div class="risk-card">
                    <h4>Percentiles</h4>
                    <p><strong>5th:</strong> {analysis['percentile_5']:.2%}</p>
                    <p><strong>25th:</strong> {analysis['percentile_25']:.2%}</p>
                    <p><strong>75th:</strong> {analysis['percentile_75']:.2%}</p>
                    <p><strong>95th:</strong> {analysis['percentile_95']:.2%}</p>
                </div>
                <div class="risk-card">
                    <h4>Probabilities</h4>
                    <p><strong>Gain:</strong> <span class="positive">{analysis['prob_gain']:.1%}</span></p>
                    <p><strong>Loss:</strong> <span class="negative">{analysis['prob_loss']:.1%}</span></p>
                </div>
            </div>
            
            <h2>📈 Simulation Results</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_b64}" alt="Monte Carlo Simulation Charts" style="max-width: 100%; height: auto;">
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background-color: #e8f6f3; border-radius: 8px;">
                <h3>💡 Key Insights</h3>
                <ul>
                    <li><strong>Expected Portfolio Growth:</strong> Based on {analysis['num_simulations']:,} simulations, your portfolio has a {analysis['prob_gain']:.1%} chance of gaining value over {config.get('SIMULATION_DAYS', 'N/A')} days.</li>
                    <li><strong>Risk Assessment:</strong> There's a 5% chance your portfolio could lose more than {abs(analysis['var_95']):.1%} of its value (VaR 95%).</li>
                    <li><strong>Return Expectation:</strong> The median expected return is {analysis['median_return']:.2%}, with potential gains up to {analysis['max_return']:.1%}.</li>
                    <li><strong>Volatility:</strong> Portfolio returns have a standard deviation of {analysis['std_return']:.2%}, indicating {'high' if analysis['std_return'] > 0.2 else 'moderate' if analysis['std_return'] > 0.1 else 'low'} volatility.</li>
                </ul>
            </div>
            
            <!-- Separator between Monte Carlo and Historical Performance -->
            <hr style="margin: 40px 0; border: 2px solid #3498db; opacity: 0.3;">
            
            {generate_historical_performance_html(historical_performance)}
            
            <div style="margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 0.9em;">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Monte Carlo Simulation</p>
                <p><em>Note: Past performance does not guarantee future results. This simulation is for educational purposes only.</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """Main execution function"""
    print("🎲 Monte Carlo Portfolio Simulation")
    print("=" * 50)
    
    # Load configuration from .env file
    config = load_env_file()
    
    # Get configuration parameters
    data_period = config.get('DATA_PERIOD', '3y')
    stock_symbols = config.get('STOCK_SYMBOLS', 'AAPL,MSFT,GOOGL').split(',')
    portfolio_holdings_str = config.get('PORTFOLIO_HOLDINGS', '')
    num_simulations = int(config.get('MONTE_CARLO_SIMULATIONS', '10000'))
    simulation_days = int(config.get('SIMULATION_DAYS', '252'))
    initial_cash = float(config.get('INITIAL_CASH', '0'))
    html_filename = 'montecarlo.html'  # Fixed filename for Monte Carlo reports
    
    # Parse portfolio holdings
    portfolio_holdings = parse_portfolio_holdings(portfolio_holdings_str)
    
    if not portfolio_holdings:
        print("❌ No valid portfolio holdings found in .env file")
        print("   Please set PORTFOLIO_HOLDINGS in format: TICKER:SHARES,TICKER:SHARES")
        return
    
    print(f"📋 Configuration:")
    print(f"   • Data Period: {data_period}")
    print(f"   • Portfolio Holdings: {len(portfolio_holdings)} stocks")
    print(f"   • Monte Carlo Simulations: {num_simulations:,}")
    print(f"   • Simulation Days: {simulation_days} ({simulation_days/252:.2f} years)")
    
    # Get stock symbols from portfolio
    symbols = list(portfolio_holdings.keys())
    
    # Fetch stock data
    price_data, failed_symbols = get_stock_data(symbols, data_period)
    
    if price_data is None:
        print("❌ Could not fetch any stock data")
        return
    
    # Calculate daily returns (much simpler and more intuitive than rolling returns)
    returns_data = calculate_daily_returns(price_data)
    
    if not returns_data:
        print("❌ Could not calculate returns for any stocks")
        return
    
    # Get current prices (last available price)
    current_prices = {}
    for symbol in price_data.columns:
        current_prices[symbol] = price_data[symbol].iloc[-1]
    
    # Run Monte Carlo simulation with daily returns
    simulation_data = monte_carlo_simulation(
        returns_data, portfolio_holdings, current_prices,
        num_simulations, simulation_days, initial_cash
    )
    
    if not simulation_data:
        print("❌ Monte Carlo simulation failed")
        return
    
    # Analyze results
    analysis = analyze_simulation_results(simulation_data, config)
    
    # Calculate historical performance
    historical_performance = calculate_historical_performance(portfolio_holdings, current_prices)
    
    # Create charts
    chart_b64, _ = create_simulation_charts(simulation_data, analysis)
    
    # Generate HTML report
    html_content = generate_html_report(simulation_data, analysis, chart_b64, config, historical_performance)
    
    # Save HTML report
    try:
        file_path = Path(html_filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📄 Monte Carlo simulation report saved as: {file_path.absolute()}")
        print(f"🌐 Open this file in your web browser to view the results")
        
        # Print summary to console
        print(f"\n📊 SIMULATION SUMMARY")
        print(f"=" * 50)
        print(f"Initial Portfolio Value: ${analysis['initial_value']:,.2f}")
        print(f"Expected Final Value: ${analysis['mean_final_value']:,.2f}")
        print(f"Expected Return: {analysis['mean_return']:.2%}")
        print(f"Standard Deviation: {analysis['std_return']:.2%}")
        print(f"Probability of Gain: {analysis['prob_gain']:.1%}")
        print(f"Value at Risk (95%): {analysis['var_95']:.2%}")
        print(f"Value at Risk (99%): {analysis['var_99']:.2%}")
        
    except Exception as e:
        print(f"❌ Error saving HTML report: {e}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
