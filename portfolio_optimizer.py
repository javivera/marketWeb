#!/usr/bin/env python3
"""
Portfolio Optimizer using Monte Carlo Simulation
===============================================
Finds the optimal 5-stock portfolio from a larger list of candidates
by maximizing returns while minimizing volatility (Sharpe ratio optimization).
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
from itertools import combinations
import json

warnings.filterwarnings('ignore')

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

def fetch_current_prices(symbols):
    """Fetch current stock prices for portfolio cost calculation"""
    print(f"💰 Fetching current prices for portfolio cost calculation...")
    current_prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get current price (latest close)
            hist = ticker.history(period='1d')
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                current_prices[symbol] = current_price
                print(f"    💲 {symbol}: ${current_price:.2f}")
            else:
                print(f"    ❌ {symbol}: No current price data")
        except Exception as e:
            print(f"    ❌ {symbol}: Error fetching price - {str(e)}")
    
    return current_prices

def calculate_portfolio_cost(portfolio_symbols, current_prices, shares_per_stock=100):
    """Calculate the total cost to buy a portfolio with specified shares per stock"""
    total_cost = 0
    cost_breakdown = {}
    
    for symbol in portfolio_symbols:
        if symbol in current_prices:
            stock_cost = current_prices[symbol] * shares_per_stock
            cost_breakdown[symbol] = stock_cost
            total_cost += stock_cost
        else:
            cost_breakdown[symbol] = 0
    
    return total_cost, cost_breakdown

def fetch_stock_data(symbols, period='3y'):
    """Fetch historical stock data"""
    print(f"📊 Fetching {period} of data for {len(symbols)} candidate stocks...")
    successful_data = {}
    
    for symbol in symbols:
        try:
            print(f"  • Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 50:  # Need sufficient data
                successful_data[symbol] = hist['Close']
                print(f"    ✅ {symbol}: {len(hist)} trading days")
            else:
                print(f"    ❌ {symbol}: Insufficient data ({len(hist)} days)")
                
        except Exception as e:
            print(f"    ❌ {symbol}: Error - {str(e)}")
            continue
    
    if not successful_data:
        print("❌ No valid stock data retrieved!")
        return None
    
    return pd.DataFrame(successful_data)
    """Fetch historical stock data"""
    print(f"📊 Fetching {period} of data for {len(symbols)} candidate stocks...")
    successful_data = {}
    
    for symbol in symbols:
        try:
            print(f"  • Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 50:  # Need sufficient data
                successful_data[symbol] = hist['Close']
                print(f"    ✅ {symbol}: {len(hist)} trading days")
            else:
                print(f"    ❌ {symbol}: Insufficient data ({len(hist)} days)")
                
        except Exception as e:
            print(f"    ❌ {symbol}: Error - {str(e)}")
            continue
    
    if not successful_data:
        print("❌ No valid stock data retrieved!")
        return None
    
    return pd.DataFrame(successful_data)

def calculate_daily_returns(price_data):
    """Calculate daily returns for all stocks"""
    print(f"📈 Calculating daily returns...")
    returns_data = price_data.pct_change().dropna()
    
    for symbol in returns_data.columns:
        print(f"    ✅ {symbol}: {len(returns_data[symbol])} daily returns")
    
    return returns_data

def monte_carlo_portfolio_analysis(returns_data, portfolio_symbols, current_prices, num_simulations=1000, simulation_days=252, shares_per_stock=100):
    """
    Run Monte Carlo simulation for a specific portfolio combination
    Returns expected return, volatility, Sharpe ratio, risk metrics, portfolio cost, and correlation
    """
    
    # Filter returns for this portfolio
    portfolio_returns = returns_data[portfolio_symbols]
    
    # Calculate average pairwise correlation
    correlation_matrix = portfolio_returns.corr()
    # Get upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlations = correlation_matrix.values[mask]
    avg_correlation = np.mean(correlations)
    
    # Equal weights
    weights = np.array([1/len(portfolio_symbols)] * len(portfolio_symbols))
    
    # Calculate portfolio cost
    portfolio_cost, cost_breakdown = calculate_portfolio_cost(portfolio_symbols, current_prices, shares_per_stock)
    
    # Calculate portfolio statistics
    mean_returns = portfolio_returns.mean().values
    cov_matrix = portfolio_returns.cov().values
    
    # Add numerical stability
    cov_matrix += np.eye(len(cov_matrix)) * 1e-8
    
    # Monte Carlo simulation
    final_returns = []
    max_drawdowns = []
    
    for _ in range(num_simulations):
        # Generate random returns for simulation period
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, simulation_days)
        
        # Calculate daily portfolio returns
        daily_portfolio_returns = np.dot(random_returns, weights)
        
        # Calculate cumulative portfolio value over time
        cumulative_values = np.cumprod(1 + daily_portfolio_returns)
        
        # Calculate maximum drawdown for this simulation
        peak = np.maximum.accumulate(cumulative_values)
        drawdown = (cumulative_values - peak) / peak
        max_drawdown = np.min(drawdown)
        max_drawdowns.append(max_drawdown)
        
        # Calculate final return
        final_return = cumulative_values[-1] - 1
        final_returns.append(final_return)
    
    # Calculate statistics
    expected_return = np.mean(final_returns)
    volatility = np.std(final_returns)
    best_case = np.max(final_returns)
    worst_case = np.min(final_returns)
    mean_max_drawdown = np.mean(max_drawdowns)
    
    # Annualize
    periods_per_year = 252 / simulation_days
    annualized_return = ((1 + expected_return) ** periods_per_year) - 1
    annualized_volatility = volatility * np.sqrt(periods_per_year)
    annualized_best_case = ((1 + best_case) ** periods_per_year) - 1
    annualized_worst_case = ((1 + worst_case) ** periods_per_year) - 1
    
    # Sharpe ratio (assuming 4.5% risk-free rate)
    risk_free_rate = 0.045
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    return {
        'expected_return': expected_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'best_case': best_case,
        'worst_case': worst_case,
        'annualized_best_case': annualized_best_case,
        'annualized_worst_case': annualized_worst_case,
        'mean_max_drawdown': mean_max_drawdown,
        'avg_correlation': avg_correlation,
        'portfolio_cost': portfolio_cost,
        'cost_breakdown': cost_breakdown,
        'symbols': portfolio_symbols
    }

def optimize_portfolio(returns_data, current_prices, target_stocks=5, num_simulations=1000, simulation_days=252, random_combinations=4000, shares_per_stock=100, max_budget=70000):
    """
    Find the optimal portfolio combination using Monte Carlo simulation
    Uses random sampling instead of testing all combinations for efficiency
    Only simulates portfolios within the specified budget
    """
    available_stocks = list(returns_data.columns)
    total_possible = len(list(combinations(available_stocks, target_stocks)))
    
    print(f"🎯 Optimizing portfolio selection...")
    print(f"   • Candidate stocks: {len(available_stocks)}")
    print(f"   • Target portfolio size: {target_stocks} stocks")
    print(f"   • Shares per stock: {shares_per_stock}")
    print(f"   • Maximum budget filter: ${max_budget:,.2f}")
    print(f"   • Total possible combinations: {total_possible:,}")
    print(f"   • Random combinations to test: {random_combinations:,}")
    print(f"   • Monte Carlo simulations per combination: {num_simulations:,}")
    print(f"   • Simulation period: {simulation_days} days")
    
    # Use random sampling approach
    print(f"   • Using random sampling with budget filtering")
    
    results = []
    tested_combinations = set()
    skipped_expensive = 0
    
    # Generate random combinations
    for i in range(random_combinations):
        # Generate a random combination
        combo = tuple(sorted(np.random.choice(available_stocks, target_stocks, replace=False)))
        
        # Skip if we've already tested this combination
        if combo in tested_combinations:
            continue
        
        tested_combinations.add(combo)
        
        try:
            # Check portfolio cost BEFORE running expensive Monte Carlo simulation
            portfolio_cost, cost_breakdown = calculate_portfolio_cost(list(combo), current_prices, shares_per_stock)
            
            # Skip if portfolio exceeds budget
            if portfolio_cost > max_budget:
                skipped_expensive += 1
                continue
            
            # Only run Monte Carlo if within budget
            result = monte_carlo_portfolio_analysis(
                returns_data, list(combo), current_prices, num_simulations, simulation_days, shares_per_stock
            )
            results.append(result)
            
            if (i + 1) % 500 == 0:
                print(f"   • Processed {i + 1:,}/{random_combinations:,} combinations (skipped {skipped_expensive:,} over budget)...")
                
        except Exception as e:
            print(f"   ⚠️  Error with combination {combo}: {str(e)}")
            continue
    
    if not results:
        print("❌ No valid portfolio combinations found within budget!")
        return None
    
    # Sort by Sharpe ratio (higher is better)
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    print(f"✅ Optimization complete!")
    print(f"   • Analyzed {len(results):,} portfolios within budget")
    print(f"   • Skipped {skipped_expensive:,} portfolios over ${max_budget:,.2f} budget")
    print(f"   • Coverage: {len(results)/total_possible*100:.2f}% of all possible combinations")
    
    # Update both console and HTML to show top 50
    print(f"\n{'='*80}")
    print("🏆 TOP 50 OPTIMIZED PORTFOLIOS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results[:50]):  # Show top 50
        stocks_str = ', '.join(result['symbols'])
        print(f"#{i+1}: {stocks_str}")
        print(f"     Return: {result['annualized_return']:.2%} | Volatility: {result['annualized_volatility']:.2%} | Sharpe: {result['sharpe_ratio']:.3f}")
        print(f"     Best Case: {result['annualized_best_case']:.2%} | Worst Case: {result['annualized_worst_case']:.2%} | Avg Max Drawdown: {result['mean_max_drawdown']:.2%}")
        print(f"     Portfolio Cost (100 shares each): ${result['portfolio_cost']:,.2f} | Avg Correlation: {result['avg_correlation']:.3f}")
        print()
    
    # Also ensure the results list passed to HTML has all portfolios
    return results  # Return all results, not just top 10

def generate_optimization_report(results, config, returns_data, max_budget, shares_per_stock):
    """Generate HTML report for portfolio optimization"""
    
    if not results:
        return "<html><body><h1>No optimization results to display</h1></body></html>"
    
    # Get top 10 portfolios
    top_portfolios = results[:10]
    
    # Create performance comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sharpe Ratio comparison
    sharpe_ratios = [p['sharpe_ratio'] for p in top_portfolios[:10]]
    portfolio_names = [f"P{i+1}" for i in range(len(sharpe_ratios))]
    
    ax1.bar(portfolio_names, sharpe_ratios, color='skyblue', edgecolor='navy')
    ax1.set_title('Top 10 Portfolios - Sharpe Ratio Comparison')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Return vs Risk scatter with drawdown info
    returns = [p['annualized_return'] for p in top_portfolios[:15]]
    risks = [p['annualized_volatility'] for p in top_portfolios[:15]]
    drawdowns = [abs(p['mean_max_drawdown']) * 1000 for p in top_portfolios[:15]]  # Scale for visibility
    
    scatter = ax2.scatter(risks, returns, c=[p['sharpe_ratio'] for p in top_portfolios[:15]], 
                         s=drawdowns, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Annualized Volatility')
    ax2.set_ylabel('Annualized Return')
    ax2.set_title('Risk-Return Profile (Color=Sharpe, Size=Avg Max Drawdown)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Sharpe Ratio')
    
    # 3. Risk Metrics Comparison
    top_5_portfolios = top_portfolios[:5]
    x_pos = np.arange(len(top_5_portfolios))
    width = 0.25
    
    best_cases = [p['annualized_best_case'] * 100 for p in top_5_portfolios]
    worst_cases = [p['annualized_worst_case'] * 100 for p in top_5_portfolios]
    drawdowns_pct = [p['mean_max_drawdown'] * 100 for p in top_5_portfolios]
    
    ax3.bar(x_pos - width, best_cases, width, label='Best Case', color='green', alpha=0.7)
    ax3.bar(x_pos, worst_cases, width, label='Worst Case', color='red', alpha=0.7)
    ax3.bar(x_pos + width, drawdowns_pct, width, label='Avg Max Drawdown', color='orange', alpha=0.7)
    
    ax3.set_xlabel('Portfolio Rank')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Risk Metrics - Top 5 Portfolios')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'#{i+1}' for i in range(len(top_5_portfolios))])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Correlation heatmap of optimal portfolio
    optimal_stocks = top_portfolios[0]['symbols']
    correlation_matrix = returns_data[optimal_stocks].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax4, fmt='.2f')
    ax4.set_title('Correlation Matrix - Optimal Portfolio')
    
    plt.tight_layout()
    
    # Convert chart to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Portfolio Optimization Results</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: rgba(255, 255, 255, 0.9);
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                position: relative;
                overflow-x: hidden;
            }}
            
            body::before {{
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: 
                    radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 200, 255, 0.1) 0%, transparent 50%);
                pointer-events: none;
                z-index: 0;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1),
                    inset 0 -1px 0 rgba(255, 255, 255, 0.05);
                position: relative;
                z-index: 1;
            }}
            
            h1 {{
                color: rgba(255, 255, 255, 0.95);
                text-align: center;
                border-bottom: 3px solid rgba(120, 200, 255, 0.6);
                padding-bottom: 15px;
                font-size: 3em;
                font-weight: 300;
                letter-spacing: -1px;
                text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
            }}
            
            h2 {{
                color: rgba(255, 255, 255, 0.9);
                border-left: 4px solid rgba(46, 204, 113, 0.8);
                padding-left: 20px;
                margin-top: 40px;
                font-weight: 400;
                text-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
            }}
            
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 25px;
                margin: 30px 0;
            }}
            
            .summary-card {{
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .summary-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, 
                    rgba(120, 119, 198, 0.8), 
                    rgba(255, 119, 198, 0.6), 
                    rgba(120, 200, 255, 0.8));
            }}
            
            .summary-card:hover {{
                background: rgba(255, 255, 255, 0.12);
                border-color: rgba(255, 255, 255, 0.2);
                transform: translateY(-2px);
            }}
            
            .optimal-portfolio {{
                background: rgba(255, 215, 0, 0.1) !important;
                border: 1px solid rgba(255, 215, 0, 0.3) !important;
                box-shadow: 
                    0 8px 32px rgba(255, 215, 0, 0.1),
                    inset 0 1px 0 rgba(255, 215, 0, 0.2) !important;
            }}
            
            .optimal-portfolio::before {{
                background: linear-gradient(90deg, 
                    rgba(255, 215, 0, 0.8), 
                    rgba(255, 193, 7, 0.6)) !important;
            }}
            
            .value {{
                font-weight: 500;
                font-size: 1.2em;
                color: rgba(120, 200, 255, 0.9);
                text-shadow: 0 0 10px rgba(120, 200, 255, 0.3);
            }}
            
            .positive {{ 
                color: rgba(46, 204, 113, 0.9) !important; 
                text-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
            }}
            
            .negative {{ 
                color: rgba(231, 76, 60, 0.9) !important; 
                text-shadow: 0 0 10px rgba(231, 76, 60, 0.3);
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            th, td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.8);
            }}
            
            th {{
                background: rgba(120, 200, 255, 0.2);
                color: rgba(255, 255, 255, 0.95);
                font-weight: 500;
                cursor: pointer;
                user-select: none;
                position: relative;
                transition: all 0.3s ease;
                text-shadow: 0 0 10px rgba(120, 200, 255, 0.3);
            }}
            
            th:hover {{
                background: rgba(120, 200, 255, 0.3);
            }}
            
            th.sortable::after {{
                content: ' ↕️';
                font-size: 0.8em;
                opacity: 0.7;
            }}
            
            th.sort-asc::after {{
                content: ' ↑';
                color: rgba(255, 215, 0, 0.9);
                font-weight: bold;
                text-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
            }}
            
            th.sort-desc::after {{
                content: ' ↓';
                color: rgba(255, 215, 0, 0.9);
                font-weight: bold;
                text-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
            }}
            
            tr:nth-child(even) {{
                background: rgba(255, 255, 255, 0.03);
            }}
            
            tr:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            
            .chart-container {{
                text-align: center;
                margin: 40px 0;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }}
            
            .chart-container img {{
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }}
            
            .stock-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin: 15px 0;
            }}
            
            .stock-tag {{
                background: rgba(120, 200, 255, 0.2);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(120, 200, 255, 0.3);
                color: rgba(255, 255, 255, 0.9);
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 2px 10px rgba(120, 200, 255, 0.1);
            }}
            
            .stock-tag:hover {{
                background: rgba(120, 200, 255, 0.3);
                transform: translateY(-1px);
                box-shadow: 0 4px 15px rgba(120, 200, 255, 0.2);
            }}
            
            .sort-info {{
                background: rgba(46, 204, 113, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(46, 204, 113, 0.2);
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                font-size: 0.9em;
                color: rgba(255, 255, 255, 0.8);
                box-shadow: 0 4px 20px rgba(46, 204, 113, 0.1);
            }}
        </style>
        <script>
            function sortTable(columnIndex, headerElement) {{
                const table = document.querySelector('table tbody');
                const rows = Array.from(table.querySelectorAll('tr'));
                const isNumeric = columnIndex > 1; // Columns 2+ are numeric
                
                // Determine sort direction
                const currentSort = headerElement.classList.contains('sort-asc') ? 'asc' : 
                                  headerElement.classList.contains('sort-desc') ? 'desc' : 'none';
                const newSort = currentSort === 'asc' ? 'desc' : 'asc';
                
                // Clear all sort classes
                document.querySelectorAll('th').forEach(th => {{
                    th.classList.remove('sort-asc', 'sort-desc');
                }});
                
                // Add new sort class
                headerElement.classList.add(`sort-${{newSort}}`);
                
                // Sort rows
                rows.sort((a, b) => {{
                    let aVal = a.cells[columnIndex].textContent.trim();
                    let bVal = b.cells[columnIndex].textContent.trim();
                    
                    if (isNumeric) {{
                        // Extract numeric value (remove %, commas, etc.)
                        aVal = parseFloat(aVal.replace(/[%,]/g, ''));
                        bVal = parseFloat(bVal.replace(/[%,]/g, ''));
                        
                        if (newSort === 'asc') {{
                            return aVal - bVal;
                        }} else {{
                            return bVal - aVal;
                        }}
                    }} else {{
                        // String comparison
                        if (newSort === 'asc') {{
                            return aVal.localeCompare(bVal);
                        }} else {{
                            return bVal.localeCompare(aVal);
                        }}
                    }}
                }});
                
                // Re-append sorted rows
                rows.forEach(row => table.appendChild(row));
                
                // Update rank column
                rows.forEach((row, index) => {{
                    row.cells[0].innerHTML = `<strong>#${{index + 1}}</strong>`;
                }});
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Portfolio Optimization Results</h1>
            
            <div class="summary-grid">
                <div class="summary-card optimal-portfolio">
                    <h3>🏆 OPTIMAL PORTFOLIO</h3>
                    <p><strong>Expected Return:</strong> <span class="value">{top_portfolios[0]['annualized_return']:.2%}</span></p>
                    <p><strong>Volatility:</strong> <span class="value">{top_portfolios[0]['annualized_volatility']:.2%}</span></p>
                    <p><strong>Sharpe Ratio:</strong> <span class="value">{top_portfolios[0]['sharpe_ratio']:.3f}</span></p>
                    <p><strong>Avg Correlation:</strong> <span class="value">{top_portfolios[0]['avg_correlation']:.3f}</span></p>
                    <p><strong>Best Case Return:</strong> <span class="value">{top_portfolios[0]['annualized_best_case']:.2%}</span></p>
                    <p><strong>Worst Case Return:</strong> <span class="value">{top_portfolios[0]['annualized_worst_case']:.2%}</span></p>
                    <p><strong>Avg Max Drawdown:</strong> <span class="value">{top_portfolios[0]['mean_max_drawdown']:.2%}</span></p>
                    <p><strong>Portfolio Cost (100 shares each):</strong> <span class="value">${top_portfolios[0]['portfolio_cost']:,.2f}</span></p>
                    <div class="stock-list">
                        {''.join([f'<span class="stock-tag">{stock}</span>' for stock in top_portfolios[0]['symbols']])}
                    </div>
                </div>
                
                <div class="summary-card">
                    <h3>📊 Analysis Summary</h3>
                    <p><strong>Random Combinations Tested:</strong> <span class="value">{len(results):,}</span></p>
                    <p><strong>Target Portfolio Size:</strong> <span class="value">5 stocks</span></p>
                    <p><strong>Budget Constraint:</strong> <span class="value">${max_budget:,.2f}</span></p>
                    <p><strong>Optimization Metric:</strong> <span class="value">Sharpe Ratio</span></p>
                    <p><strong>Risk-Free Rate:</strong> <span class="value">4.5%</span></p>
                    <p><strong>Sampling Method:</strong> <span class="value">Random</span></p>
                </div>
            </div>
            
            <h2>📈 Optimization Charts</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_b64}" alt="Portfolio Optimization Charts" style="max-width: 100%; height: auto;">
            </div>
            
            <h2>🏆 Top 10 Optimized Portfolios</h2>
            <div class="sort-info">
                💡 <strong>Click on column headers to sort the table.</strong> Click again to reverse the sort order.
                <br>📊 <strong>Avg Correlation:</strong> Lower values (closer to 0) indicate better diversification.
            </div>
            <table>
                <thead>
                    <tr>
                        <th onclick="sortTable(0, this)" class="sortable">Rank</th>
                        <th onclick="sortTable(1, this)" class="sortable">Portfolio Stocks</th>
                        <th onclick="sortTable(2, this)" class="sortable">Expected Return</th>
                        <th onclick="sortTable(3, this)" class="sortable">Volatility</th>
                        <th onclick="sortTable(4, this)" class="sortable">Sharpe Ratio</th>
                        <th onclick="sortTable(5, this)" class="sortable">Best Case</th>
                        <th onclick="sortTable(6, this)" class="sortable">Worst Case</th>
                        <th onclick="sortTable(7, this)" class="sortable">Avg Max Drawdown</th>
                        <th onclick="sortTable(8, this)" class="sortable">Avg Correlation</th>
                        <th onclick="sortTable(9, this)" class="sortable">Portfolio Cost ({shares_per_stock} shares each)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Generate table rows for top 50 portfolios
    for i, result in enumerate(results[:50]):  # Show top 50
        stocks_str = ', '.join(result['symbols'])
        best_case_color = "positive" if result['annualized_best_case'] > 0 else "negative"  
        worst_case_color = "positive" if result['annualized_worst_case'] > 0 else "negative"
        drawdown_color = "negative"  # Drawdowns are always negative/bad
        
        # Color correlation: lower is better (more diversified)
        correlation_color = "positive" if result['avg_correlation'] < 0.3 else "negative" if result['avg_correlation'] > 0.7 else ""
        
        html_content += f"""
                    <tr>
                        <td><strong>#{i}</strong></td>
                        <td>{stocks_str}</td>
                        <td><span class="value positive">{result['annualized_return']:.2%}</span></td>
                        <td><span class="value">{result['annualized_volatility']:.2%}</span></td>
                        <td><span class="value positive">{result['sharpe_ratio']:.3f}</span></td>
                        <td><span class="value {best_case_color}">{result['annualized_best_case']:.2%}</span></td>
                        <td><span class="value {worst_case_color}">{result['annualized_worst_case']:.2%}</span></td>
                        <td><span class="value {drawdown_color}">{result['mean_max_drawdown']:.2%}</span></td>
                        <td><span class="value {correlation_color}">{result['avg_correlation']:.3f}</span></td>
                        <td><span class="value">${result['portfolio_cost']:,.2f}</span></td>
                    </tr>
        """
    
    html_content += f"""
                </tbody>
            </table>
            
            <div style="margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 0.9em;">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Portfolio Optimization Analysis</p>
                <p><em>This analysis is for educational purposes only. Past performance does not guarantee future results.</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """Main execution function"""
    print("🎯 Portfolio Optimizer using Monte Carlo Simulation")
    print("=" * 60)
    
    # Load configuration
    config = load_env_file()
    
    # Get configuration parameters
    data_period = config.get('DATA_PERIOD', '3y')
    candidate_stocks_str = config.get('OPTIMIZER_STOCKS', 'AAPL,MSFT,GOOG,AMZN,TSLA,META,NVDA,AMD,SPY,TSM,SOXL,QQQ,EEM,EWZ,VTI,IWM,FXI,PLTR')
    candidate_stocks = [s.strip() for s in candidate_stocks_str.split(',')]
    
    num_simulations = int(config.get('OPTIMIZER_SIMULATIONS', '500'))  # Fewer sims per combination due to volume
    simulation_days = int(config.get('SIMULATION_DAYS', '252'))
    target_stocks = int(config.get('TARGET_PORTFOLIO_SIZE', '5'))
    random_combinations = int(config.get('RANDOM_COMBINATIONS', '4000'))
    shares_per_stock = int(config.get('SHARES_PER_STOCK', '100'))
    max_budget = float(config.get('MAX_PORTFOLIO_BUDGET', '70000'))
    
    print(f"📋 Configuration:")
    print(f"   • Data Period: {data_period}")
    print(f"   • Candidate Stocks: {len(candidate_stocks)} stocks")
    print(f"   • Target Portfolio Size: {target_stocks} stocks")
    print(f"   • Maximum Budget: ${max_budget:,.2f}")
    print(f"   • Shares per Stock: {shares_per_stock}")
    print(f"   • Random Combinations to Test: {random_combinations:,}")
    print(f"   • Monte Carlo Simulations per Portfolio: {num_simulations:,}")
    print(f"   • Simulation Days: {simulation_days}")
    
    # Fetch stock data
    price_data = fetch_stock_data(candidate_stocks, data_period)
    if price_data is None:
        return
    
    # Calculate returns
    returns_data = calculate_daily_returns(price_data)
    
    # Get unique stocks from returns data to fetch current prices
    unique_stocks = list(returns_data.columns)
    current_prices = fetch_current_prices(unique_stocks)
    
    # Run optimization
    results = optimize_portfolio(
        returns_data,
        current_prices,
        target_stocks,
        num_simulations,
        simulation_days,
        random_combinations,
        shares_per_stock,
        max_budget,
    )
    if not results:
        print("\n⚠️  No valid portfolio combinations found within budget.")
        if shares_per_stock > 1:
            print("   Retrying automatically with SHARES_PER_STOCK=1...")
            results = optimize_portfolio(
                returns_data,
                current_prices,
                target_stocks,
                num_simulations,
                simulation_days,
                random_combinations,
                1,
                max_budget,
            )
            if results:
                shares_per_stock = 1
        if not results:
            # Compute a rough minimum feasible cost for guidance
            # Take the cheapest target_stocks among candidate current_prices
            sorted_prices = sorted([price for _, price in current_prices.items() if price is not None])
            if len(sorted_prices) >= target_stocks:
                min_cost_1_share = sum(sorted_prices[:target_stocks])
                print(f"   ℹ️ Minimum estimated cost with 1 share each: ${min_cost_1_share:,.2f}")
            print("\n❌ Still no valid portfolios. Try one or more of these:")
            print("   • Decrease SHARES_PER_STOCK (e.g., 1 or 5) in .env")
            print("   • Increase MAX_PORTFOLIO_BUDGET in .env")
            print("   • Remove very expensive tickers from OPTIMIZER_STOCKS")
            return
    
    # Print top results
    print(f"\\n🏆 TOP 5 OPTIMIZED PORTFOLIOS:")
    print("=" * 90)
    for i, portfolio in enumerate(results[:5], 1):
        stocks_str = ', '.join(portfolio['symbols'])
        print(f"#{i}: {stocks_str}")
        print(f"     Return: {portfolio['annualized_return']:.2%} | Volatility: {portfolio['annualized_volatility']:.2%} | Sharpe: {portfolio['sharpe_ratio']:.3f}")
        print(f"     Best Case: {portfolio['annualized_best_case']:.2%} | Worst Case: {portfolio['annualized_worst_case']:.2%} | Avg Max Drawdown: {portfolio['mean_max_drawdown']:.2%}")
        print(f"     Portfolio Cost (100 shares each): ${portfolio['portfolio_cost']:,.2f} | Avg Correlation: {portfolio['avg_correlation']:.3f}")
        print()
    
    # Generate HTML report
    html_content = generate_optimization_report(results, config, returns_data, max_budget, shares_per_stock)
    
    # Save HTML report
    html_filename = 'portfolio_optimization.html'
    file_path = Path(html_filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\\n📄 Portfolio optimization report saved as: {file_path.absolute()}")
    print("🌐 Open this file in your web browser to view the results")

def run_analysis_once():
    """Run the portfolio optimization analysis once"""
    warnings.filterwarnings('ignore')
    main()

def run_scheduler():
    """Run the analysis on a schedule"""
    config = load_env_file()
    interval_hours = int(config.get('SCHEDULE_INTERVAL_HOURS', '4'))
    
    print(f"🕐 Starting automatic portfolio optimization scheduler")
    print(f"📅 Analysis will run every {interval_hours} hours")
    print(f"⏰ Next run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🛑 Press Ctrl+C to stop the scheduler")
    print("=" * 60)
    
    try:
        while True:
            # Run analysis
            print(f"\n🚀 Starting scheduled portfolio optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            run_analysis_once()
            
            # Calculate next run time
            next_run = datetime.now() + timedelta(hours=interval_hours)
            print(f"⏰ Next analysis scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💤 Sleeping for {interval_hours} hours...")
            
            # Sleep for the specified interval
            time.sleep(interval_hours * 3600)  # Convert hours to seconds
            
    except KeyboardInterrupt:
        print(f"\n🛑 Scheduler stopped by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("👋 Goodbye!")

def main_with_args():
    """Main function with command line argument handling"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['schedule', 'scheduler', '--schedule', '-s']:
            run_scheduler()
        elif sys.argv[1].lower() in ['once', '--once', '-o']:
            run_analysis_once()
        elif sys.argv[1].lower() in ['help', '--help', '-h']:
            print("Portfolio Optimizer Tool")
            print("=" * 40)
            print("Usage:")
            print("  python3 portfolio_optimizer.py          # Run once")
            print("  python3 portfolio_optimizer.py once     # Run once")
            print("  python3 portfolio_optimizer.py schedule # Run every 4 hours")
            print("  python3 portfolio_optimizer.py help     # Show this help")
            print("\nConfiguration:")
            print("  Edit .env file to change settings:")
            print("  - SCHEDULE_INTERVAL_HOURS: How often to run (default: 4)")
            print("  - RANDOM_COMBINATIONS: Number of portfolios to test (default: 15000)")
            print("  - MAX_PORTFOLIO_BUDGET: Maximum budget for portfolios (default: 70000)")
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use 'help' to see available options")
    else:
        # Default behavior - run once
        run_analysis_once()

if __name__ == "__main__":
    main_with_args()
