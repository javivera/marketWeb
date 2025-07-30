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
                env_vars[key] = value
    
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

def calculate_returns(price_data, lookback_days):
    """Calculate returns for different time periods"""
    print(f"📈 Calculating {lookback_days}-day returns...")
    
    returns_data = {}
    
    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        returns = []
        
        # Calculate rolling returns
        for i in range(lookback_days, len(prices)):
            start_price = prices.iloc[i - lookback_days]
            end_price = prices.iloc[i]
            period_return = (end_price - start_price) / start_price
            returns.append(period_return)
        
        if returns:
            returns_data[symbol] = np.array(returns)
        else:
            print(f"⚠️  {symbol}: Insufficient data for {lookback_days}-day periods")
    
    return returns_data

def monte_carlo_simulation(returns_data, portfolio_holdings, current_prices, 
                         num_simulations, simulation_days, initial_cash=0):
    """
    Run Monte Carlo simulation for portfolio returns
    
    Parameters:
    - returns_data: Historical returns for each stock
    - portfolio_holdings: Dictionary of {symbol: shares}
    - current_prices: Current prices for each stock
    - num_simulations: Number of Monte Carlo runs
    - simulation_days: Number of days to simulate into the future
    - initial_cash: Additional cash in the portfolio
    """
    
    print(f"🎲 Running Monte Carlo simulation...")
    print(f"   • Simulations: {num_simulations:,}")
    print(f"   • Time horizon: {simulation_days} days")
    print(f"   • Portfolio stocks: {len(portfolio_holdings)}")
    
    # Calculate current portfolio value
    portfolio_value = initial_cash
    valid_holdings = {}
    
    for symbol, shares in portfolio_holdings.items():
        if symbol in current_prices and symbol in returns_data:
            stock_value = shares * current_prices[symbol]
            portfolio_value += stock_value
            valid_holdings[symbol] = shares
            print(f"   • {symbol}: {shares} shares @ ${current_prices[symbol]:.2f} = ${stock_value:,.2f}")
        else:
            print(f"   ⚠️  {symbol}: No data available, excluding from simulation")
    
    print(f"   • Total initial portfolio value: ${portfolio_value:,.2f}")
    
    if not valid_holdings:
        print("❌ No valid holdings found for simulation")
        return None
    
    # Prepare returns statistics for simulation
    stocks = list(valid_holdings.keys())
    returns_matrix = np.array([returns_data[stock] for stock in stocks]).T
    
    # Calculate mean returns and covariance matrix
    mean_returns = np.mean(returns_matrix, axis=0)
    cov_matrix = np.cov(returns_matrix.T)
    
    print(f"   • Mean daily returns: {mean_returns * 100}")
    print(f"   • Running {num_simulations:,} simulations...")
    
    # Run Monte Carlo simulations
    simulation_results = []
    
    for sim in range(num_simulations):
        # Generate random returns for each day
        portfolio_values = [portfolio_value]
        current_portfolio_value = portfolio_value
        
        for day in range(simulation_days):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            
            # Apply returns to each stock in the portfolio
            day_change = 0
            for i, symbol in enumerate(stocks):
                shares = valid_holdings[symbol]
                current_price = current_portfolio_value * (shares * current_prices[symbol]) / portfolio_value
                stock_change = current_price * random_returns[i]
                day_change += stock_change
            
            current_portfolio_value += day_change
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
        'stocks': stocks,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix
    }

def analyze_simulation_results(simulation_data):
    """Analyze Monte Carlo simulation results"""
    
    if not simulation_data:
        return None
    
    results = simulation_data['simulations']
    initial_value = simulation_data['initial_value']
    
    # Extract results
    final_values = [r['final_value'] for r in results]
    total_returns = [r['total_return'] for r in results]
    
    # Calculate statistics
    analysis = {
        'initial_value': initial_value,
        'num_simulations': len(results),
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'std_final_value': np.std(final_values),
        'mean_return': np.mean(total_returns),
        'median_return': np.median(total_returns),
        'std_return': np.std(total_returns),
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
    
    return analysis

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

def generate_html_report(simulation_data, analysis, chart_b64, config):
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
                    <p><strong>Time Horizon:</strong> <span class="value">{config.get('SIMULATION_DAYS', 'N/A')} days</span></p>
                    <p><strong>Lookback Period:</strong> <span class="value">{config.get('LOOKBACK_DAYS', 'N/A')} days</span></p>
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
                    <p><strong>Median Return:</strong> <span class="value {'positive' if analysis['median_return'] > 0 else 'negative'}">{analysis['median_return']:.2%}</span></p>
                    <p><strong>Std Deviation:</strong> <span class="value">{analysis['std_return']:.2%}</span></p>
                </div>
                
                <div class="summary-card">
                    <h3>⚠️ Risk Metrics</h3>
                    <p><strong>VaR (95%):</strong> <span class="value negative">{analysis['var_95']:.2%}</span></p>
                    <p><strong>VaR (99%):</strong> <span class="value negative">{analysis['var_99']:.2%}</span></p>
                    <p><strong>Prob. of Loss:</strong> <span class="value">{analysis['prob_loss']:.1%}</span></p>
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
    lookback_days = int(config.get('LOOKBACK_DAYS', '30'))
    stock_symbols = config.get('STOCK_SYMBOLS', 'AAPL,MSFT,GOOGL').split(',')
    portfolio_holdings_str = config.get('PORTFOLIO_HOLDINGS', '')
    num_simulations = int(config.get('MONTE_CARLO_SIMULATIONS', '10000'))
    simulation_days = int(config.get('SIMULATION_DAYS', '252'))
    initial_cash = float(config.get('INITIAL_CASH', '0'))
    html_filename = config.get('HTML_FILENAME', 'monte_carlo_results.html')
    
    # Parse portfolio holdings
    portfolio_holdings = parse_portfolio_holdings(portfolio_holdings_str)
    
    if not portfolio_holdings:
        print("❌ No valid portfolio holdings found in .env file")
        print("   Please set PORTFOLIO_HOLDINGS in format: TICKER:SHARES,TICKER:SHARES")
        return
    
    print(f"📋 Configuration:")
    print(f"   • Data Period: {data_period}")
    print(f"   • Lookback Days: {lookback_days}")
    print(f"   • Portfolio Holdings: {len(portfolio_holdings)} stocks")
    print(f"   • Monte Carlo Simulations: {num_simulations:,}")
    print(f"   • Simulation Days: {simulation_days}")
    
    # Get stock symbols from portfolio
    symbols = list(portfolio_holdings.keys())
    
    # Fetch stock data
    price_data, failed_symbols = get_stock_data(symbols, data_period)
    
    if price_data is None:
        print("❌ Could not fetch any stock data")
        return
    
    # Calculate returns
    returns_data = calculate_returns(price_data, lookback_days)
    
    if not returns_data:
        print("❌ Could not calculate returns for any stocks")
        return
    
    # Get current prices (last available price)
    current_prices = {}
    for symbol in price_data.columns:
        current_prices[symbol] = price_data[symbol].iloc[-1]
    
    # Run Monte Carlo simulation
    simulation_data = monte_carlo_simulation(
        returns_data, portfolio_holdings, current_prices,
        num_simulations, simulation_days, initial_cash
    )
    
    if not simulation_data:
        print("❌ Monte Carlo simulation failed")
        return
    
    # Analyze results
    analysis = analyze_simulation_results(simulation_data)
    
    # Create charts
    chart_b64, _ = create_simulation_charts(simulation_data, analysis)
    
    # Generate HTML report
    html_content = generate_html_report(simulation_data, analysis, chart_b64, config)
    
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
