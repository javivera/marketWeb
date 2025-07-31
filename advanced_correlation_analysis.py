import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import base64
import io
from pathlib import Path
import os
import time
import threading
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
                env_vars[key] = value
    
    return env_vars

def get_stock_data(symbols, period="1y"):
    """
    Fetch stock data for multiple symbols
    
    Parameters:
    - symbols: List of stock symbols
    - period: Time period ('1y', '2y', '3y', '5y')
    
    Returns:
    - DataFrame with stock prices, list of failed symbols
    """
    print(f"📊 Fetching {period} of data for {len(symbols)} stocks...")
    
    stock_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            print(f"  • Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"    ❌ No data found for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            stock_data[symbol] = data['Close']
            print(f"    ✅ {symbol}: {len(data)} trading days")
            
        except Exception as e:
            print(f"    ❌ Error fetching {symbol}: {e}")
            failed_symbols.append(symbol)
    
    if not stock_data:
        print("❌ No valid data found for any symbols")
        return None, failed_symbols
    
    # Combine into DataFrame
    price_data = pd.DataFrame(stock_data)
    
    # Remove any rows with all NaN values
    price_data = price_data.dropna(how='all')
    
    if failed_symbols:
        print(f"⚠️  Failed to fetch data for: {', '.join(failed_symbols)}")
    
    return price_data, failed_symbols

def calculate_period_returns(price_data, time_frame_days):
    """
    Calculate returns over specified time frame periods
    
    Parameters:
    - price_data: DataFrame with stock prices
    - time_frame_days: Number of days for each return calculation period
    
    Returns:
    - DataFrame with period returns
    """
    print(f"📈 Calculating {time_frame_days}-day period returns...")
    
    period_returns = {}
    
    for symbol in price_data.columns:
        prices = price_data[symbol].dropna()
        
        if len(prices) < time_frame_days + 1:
            print(f"⚠️  {symbol}: Insufficient data for {time_frame_days}-day periods")
            continue
        
        returns = []
        
        # Calculate returns for each time_frame_days period
        for i in range(time_frame_days, len(prices)):
            start_price = prices.iloc[i - time_frame_days]
            end_price = prices.iloc[i]
            period_return = (end_price / start_price - 1) * 100  # Convert to percentage
            returns.append(period_return)
        
        period_returns[symbol] = returns
    
    # Convert to DataFrame with aligned lengths
    max_length = max(len(returns) for returns in period_returns.values()) if period_returns else 0
    
    aligned_returns = {}
    for symbol, returns in period_returns.items():
        # Pad with NaN to align lengths
        padded_returns = [np.nan] * (max_length - len(returns)) + returns
        aligned_returns[symbol] = padded_returns
    
    return pd.DataFrame(aligned_returns)

def calculate_return_statistics(period_returns, symbols, time_frame_days):
    """
    Calculate comprehensive return statistics for each stock
    
    Parameters:
    - period_returns: DataFrame with period returns
    - symbols: List of stock symbols
    - time_frame_days: Time frame used for calculations
    
    Returns:
    - Dictionary with return statistics
    """
    print(f"📊 Calculating return statistics for {time_frame_days}-day periods...")
    
    stats = {}
    
    for symbol in symbols:
        if symbol not in period_returns.columns:
            continue
        
        returns = period_returns[symbol].dropna()
        
        if len(returns) == 0:
            continue
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        # Calculate statistics
        symbol_stats = {
            'total_periods': len(returns),
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'std_return': returns.std(),
            
            # Positive returns analysis
            'positive_periods': len(positive_returns),
            'positive_rate': len(positive_returns) / len(returns) * 100,
            'mean_positive_return': positive_returns.mean() if len(positive_returns) > 0 else 0,
            'max_positive_return': positive_returns.max() if len(positive_returns) > 0 else 0,
            'median_positive_return': positive_returns.median() if len(positive_returns) > 0 else 0,
            
            # Negative returns analysis
            'negative_periods': len(negative_returns),
            'negative_rate': len(negative_returns) / len(returns) * 100,
            'mean_negative_return': negative_returns.mean() if len(negative_returns) > 0 else 0,
            'max_negative_return': negative_returns.min() if len(negative_returns) > 0 else 0,  # Most negative
            'median_negative_return': negative_returns.median() if len(negative_returns) > 0 else 0,
            
            # Risk metrics
            'volatility': returns.std(),
            'downside_deviation': negative_returns.std() if len(negative_returns) > 0 else 0,
            'upside_deviation': positive_returns.std() if len(positive_returns) > 0 else 0,
            
            # Percentiles
            'percentile_5': returns.quantile(0.05),
            'percentile_25': returns.quantile(0.25),
            'percentile_75': returns.quantile(0.75),
            'percentile_95': returns.quantile(0.95),
            
            # Additional metrics
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
            'sortino_ratio': returns.mean() / negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() != 0 else 0
        }
        
        stats[symbol] = symbol_stats
    
    return stats

def calculate_correlation_matrix(period_returns):
    """Calculate correlation matrix of period returns"""
    return period_returns.corr()

def print_correlation_analysis(correlation_matrix, symbols):
    """Print correlation matrix analysis"""
    print(f"\n{'CORRELATION MATRIX ANALYSIS':-^80}")
    
    # Print correlation matrix
    print(f"\nCorrelation Matrix:")
    print("-" * 60)
    
    # Create formatted correlation matrix
    corr_df = correlation_matrix.round(3)
    print(corr_df.to_string())
    
    print(f"\n{'CORRELATION INSIGHTS':-^80}")
    
    # Find highest and lowest correlations (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    correlations = []
    
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            if symbols[i] in correlation_matrix.index and symbols[j] in correlation_matrix.columns:
                corr_value = correlation_matrix.loc[symbols[i], symbols[j]]
                if not np.isnan(corr_value):
                    correlations.append((symbols[i], symbols[j], corr_value))
    
    if correlations:
        # Sort by correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("🔗 Highest Correlations:")
        for i, (stock1, stock2, corr) in enumerate(correlations[:5]):
            print(f"  {i+1}. {stock1} - {stock2}: {corr:.3f}")
        
        print("\n🔄 Lowest Correlations:")
        correlations.sort(key=lambda x: abs(x[2]))
        for i, (stock1, stock2, corr) in enumerate(correlations[:5]):
            print(f"  {i+1}. {stock1} - {stock2}: {corr:.3f}")

def print_return_statistics_analysis(return_stats, symbols, time_frame_days, data_period):
    """Print comprehensive return statistics analysis"""
    
    print(f"\n{'RETURN STATISTICS ANALYSIS':-^80}")
    print(f"Time Frame: {time_frame_days} days | Data Period: {data_period.upper()}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*80}")
    
    # Sort stocks by mean return (highest to lowest)
    sorted_stocks = [(symbol, stats) for symbol, stats in return_stats.items()]
    sorted_stocks.sort(key=lambda x: x[1]['mean_return'], reverse=True)
    
    print(f"\n{'PERFORMANCE RANKINGS':-^80}")
    print(f"{'Rank':<5} {'Symbol':<8} {'Mean Ret':<10} {'Pos Rate':<10} {'Volatility':<12} {'Sharpe':<8}")
    print("-" * 80)
    
    for rank, (symbol, stats) in enumerate(sorted_stocks, 1):
        mean_ret = stats['mean_return']
        pos_rate = stats['positive_rate']
        volatility = stats['volatility']
        sharpe = stats['sharpe_ratio']
        
        print(f"{rank:<5} {symbol:<8} {mean_ret:<10.2f}% {pos_rate:<10.1f}% {volatility:<12.2f}% {sharpe:<8.2f}")
    
    print(f"\n{'DETAILED STATISTICS BY STOCK':-^80}")
    
    for symbol, stats in sorted_stocks:
        print(f"\n📊 {symbol} - {time_frame_days}-Day Period Analysis")
        print(f"  📈 RETURN METRICS:")
        print(f"    • Mean Return: {stats['mean_return']:.2f}%")
        print(f"    • Median Return: {stats['median_return']:.2f}%")
        print(f"    • Standard Deviation: {stats['std_return']:.2f}%")
        print(f"    • Total Periods Analyzed: {stats['total_periods']}")
        
        print(f"  ✅ POSITIVE RETURNS:")
        print(f"    • Positive Periods: {stats['positive_periods']} ({stats['positive_rate']:.1f}%)")
        print(f"    • Mean Positive Return: {stats['mean_positive_return']:.2f}%")
        print(f"    • Median Positive Return: {stats['median_positive_return']:.2f}%")
        print(f"    • Maximum Gain: {stats['max_positive_return']:.2f}%")
        
        print(f"  ❌ NEGATIVE RETURNS:")
        print(f"    • Negative Periods: {stats['negative_periods']} ({stats['negative_rate']:.1f}%)")
        print(f"    • Mean Negative Return: {stats['mean_negative_return']:.2f}%")
        print(f"    • Median Negative Return: {stats['median_negative_return']:.2f}%")
        print(f"    • Maximum Loss: {stats['max_negative_return']:.2f}%")
        
        print(f"  📊 RISK METRICS:")
        print(f"    • Volatility: {stats['volatility']:.2f}%")
        print(f"    • Downside Deviation: {stats['downside_deviation']:.2f}%")
        print(f"    • Upside Deviation: {stats['upside_deviation']:.2f}%")
        print(f"    • Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"    • Sortino Ratio: {stats['sortino_ratio']:.3f}")
        
        print(f"  📈 DISTRIBUTION:")
        print(f"    • 5th Percentile: {stats['percentile_5']:.2f}%")
        print(f"    • 25th Percentile: {stats['percentile_25']:.2f}%")
        print(f"    • 75th Percentile: {stats['percentile_75']:.2f}%")
        print(f"    • 95th Percentile: {stats['percentile_95']:.2f}%")
        print(f"    • Skewness: {stats['skewness']:.3f}")
        print(f"    • Kurtosis: {stats['kurtosis']:.3f}")
        
        # Risk assessment
        vol = stats['volatility']
        if vol > 15:
            risk_level = "🔥 VERY HIGH"
        elif vol > 10:
            risk_level = "⚡ HIGH"
        elif vol > 5:
            risk_level = "📊 MODERATE"
        else:
            risk_level = "🟢 LOW"
        
        print(f"    • Risk Level: {risk_level}")

def save_plot_to_base64():
    """Save current matplotlib plot to base64 string for HTML embedding"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()  # Close the figure to free memory
    return image_base64

def create_correlation_heatmap(correlation_matrix, title):
    """Create correlation heatmap and return base64 image"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return save_plot_to_base64()

def create_return_distribution_plots(period_returns, symbols, time_frame_days):
    """Create return distribution plots and return base64 image"""
    n_stocks = len(symbols)
    
    if n_stocks <= 4:
        rows, cols = 2, 2
    elif n_stocks <= 6:
        rows, cols = 2, 3
    elif n_stocks <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    if n_stocks == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, symbol in enumerate(symbols):
        if i >= len(axes):
            break
            
        if symbol not in period_returns.columns:
            continue
            
        returns = period_returns[symbol].dropna()
        
        if len(returns) == 0:
            axes[i].text(0.5, 0.5, f'No data\nfor {symbol}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(symbol)
            continue
        
        # Create histogram
        axes[i].hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        axes[i].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Return')
        
        axes[i].set_title(f'{symbol} - {time_frame_days}D Returns', fontweight='bold')
        axes[i].set_xlabel('Return (%)')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(symbols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Return Distribution Analysis ({time_frame_days}-Day Periods)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return save_plot_to_base64()

def create_return_comparison_chart(return_stats, symbols, time_frame_days):
    """Create comprehensive return comparison charts and return base64 image"""
    
    # Extract data for plotting
    mean_returns = [return_stats[symbol]['mean_return'] for symbol in symbols if symbol in return_stats]
    positive_rates = [return_stats[symbol]['positive_rate'] for symbol in symbols if symbol in return_stats]
    volatilities = [return_stats[symbol]['volatility'] for symbol in symbols if symbol in return_stats]
    max_gains = [return_stats[symbol]['max_positive_return'] for symbol in symbols if symbol in return_stats]
    max_losses = [return_stats[symbol]['max_negative_return'] for symbol in symbols if symbol in return_stats]
    valid_symbols = [symbol for symbol in symbols if symbol in return_stats]
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean Returns Comparison
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(valid_symbols)))
    bars1 = ax1.bar(valid_symbols, mean_returns, color=colors)
    ax1.set_title(f'Mean {time_frame_days}-Day Returns Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Return (%)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, ret in zip(bars1, mean_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., 
                height + (0.1 if height >= 0 else -0.3),
                f'{ret:.1f}%', ha='center', 
                va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 2. Positive Return Rate vs Volatility
    scatter = ax2.scatter(volatilities, positive_rates, s=100, alpha=0.7, c=colors)
    ax2.set_xlabel('Volatility (%)', fontsize=12)
    ax2.set_ylabel('Positive Return Rate (%)', fontsize=12)
    ax2.set_title('Risk vs Success Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add stock labels
    for i, symbol in enumerate(valid_symbols):
        ax2.annotate(symbol, (volatilities[i], positive_rates[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    # 3. Max Gains vs Max Losses
    ax3.scatter(max_losses, max_gains, s=100, alpha=0.7, c=colors)
    ax3.set_xlabel('Maximum Loss (%)', fontsize=12)
    ax3.set_ylabel('Maximum Gain (%)', fontsize=12)
    ax3.set_title('Extreme Returns Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add stock labels
    for i, symbol in enumerate(valid_symbols):
        ax3.annotate(symbol, (max_losses[i], max_gains[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    # 4. Volatility Comparison
    bars4 = ax4.bar(valid_symbols, volatilities, color=colors, alpha=0.7)
    ax4.set_title(f'Volatility Comparison ({time_frame_days}-Day Periods)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Volatility (%)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, vol in zip(bars4, volatilities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{vol:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Comprehensive Return Analysis ({time_frame_days}-Day Periods)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return save_plot_to_base64()

def generate_html_report(correlation_data, return_stats_text, symbols, data_period, time_frame_days, 
                        distribution_plots_b64, comparison_chart_b64):
    """Generate comprehensive HTML report with all analysis and charts"""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format correlation matrix as HTML table
    correlation_matrix = correlation_data
    correlation_html = correlation_matrix.round(3).to_html(classes='correlation-table', table_id='correlation-table')
    
    # Extract key statistics for summary
    summary_stats = []
    for symbol in symbols:
        if symbol in return_stats_text:
            stats = return_stats_text[symbol]
            summary_stats.append({
                'symbol': symbol,
                'mean_return': stats['mean_return'],
                'positive_rate': stats['positive_rate'],
                'volatility': stats['volatility'],
                'sharpe_ratio': stats['sharpe_ratio']
            })
    
    # Sort by mean return for summary table
    summary_stats.sort(key=lambda x: x['mean_return'], reverse=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Correlation & Return Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
                min-height: 100vh;
                color: #e8e9ea;
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
                padding: 30px;
                border-radius: 20px;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1),
                    inset 0 -1px 0 rgba(255, 255, 255, 0.05);
                position: relative;
                z-index: 1;
            }}
            
            p, li, div {{
                color: #e8e9ea;
            }}
            
            .header {{
                text-align: center;
                border-bottom: 2px solid rgba(120, 200, 255, 0.6);
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                color: #ffffff;
                margin: 0;
                font-size: 3em;
                font-weight: 300;
                letter-spacing: -1px;
                text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            }}
            
            .header p {{
                color: #b8bcc0;
                margin: 15px 0 0 0;
                font-size: 1.2em;
                font-weight: 300;
            }}
            
            .section {{
                margin: 50px 0;
                padding: 30px;
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            }}
            
            .section::before {{
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
            
            .section h2 {{
                color: #ffffff;
                border-bottom: 2px solid rgba(120, 200, 255, 0.6);
                padding-bottom: 15px;
                margin-top: 0;
                font-weight: 400;
                text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }}
            
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }}
            
            .summary-table th, .summary-table td {{
                padding: 12px 15px;
                text-align: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                color: #e8e9ea;
            }}
            
            .summary-table th {{
                background: rgba(120, 200, 255, 0.2);
                color: #ffffff;
                font-weight: 500;
                cursor: pointer;
                user-select: none;
                position: relative;
                transition: all 0.3s ease;
                text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }}
            
            .summary-table th:hover {{
                background: rgba(120, 200, 255, 0.3);
            }}
            
            .summary-table th::after {{
                content: ' ↕️';
                font-size: 0.7em;
                opacity: 0.7;
            }}
            
            .summary-table tr:nth-child(even) {{
                background: rgba(255, 255, 255, 0.03);
            }}
            
            .summary-table tr:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            
            .correlation-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }}
            
            .correlation-table th, .correlation-table td {{
                padding: 8px 12px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #e8e9ea;
                font-size: 0.9em;
            }}
            
            .correlation-table th {{
                background: rgba(52, 73, 94, 0.4);
                color: #ffffff;
                font-weight: 500;
                cursor: pointer;
                user-select: none;
                text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }}
            
            .correlation-table th:hover {{
                background: rgba(44, 62, 80, 0.5);
            }}
            
            .chart-container {{
                text-align: center;
                margin: 30px 0;
                padding: 25px;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }}
            
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }}
            
            .info-box {{
                background: rgba(46, 204, 113, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(46, 204, 113, 0.3);
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 4px 20px rgba(46, 204, 113, 0.1);
            }}
            
            .info-box h4 {{
                color: rgba(46, 204, 113, 1);
                margin-top: 0;
                margin-bottom: 12px;
                text-shadow: 0 1px 5px rgba(46, 204, 113, 0.5);
            }}
            
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                color: #9ca3af;
                font-weight: 300;
            }}
            
            .positive {{ 
                color: rgba(46, 204, 113, 1) !important; 
                font-weight: 500 !important;
                text-shadow: 0 1px 5px rgba(46, 204, 113, 0.5);
            }}
            
            .negative {{ 
                color: rgba(231, 76, 60, 1) !important; 
                font-weight: 500 !important;
                text-shadow: 0 1px 5px rgba(231, 76, 60, 0.5);
            }}
            
            .neutral {{ 
                color: rgba(243, 156, 18, 1) !important; 
                font-weight: 500 !important;
                text-shadow: 0 1px 5px rgba(243, 156, 18, 0.5);
            }}
            
            /* Collapsible sections */
            .collapsible {{
                background-color: #3498db;
                color: white;
                cursor: pointer;
                padding: 10px 15px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                margin: 5px 0;
                transition: 0.3s;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .collapsible:hover {{
                background-color: #2980b9;
            }}
            
            .collapsible.active {{
                background-color: #2980b9;
            }}
            
            .collapsible-content {{
                max-height: 0;
                padding: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(15px);
                border-radius: 0 0 5px 5px;
                border-left: 3px solid #3498db;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .collapsible-content.show {{
                padding: 20px;
            }}
            }}
            }}
            
            .collapsible-icon {{
                font-size: 16px;
                transition: transform 0.3s;
            }}
            
            .collapsible.active .collapsible-icon {{
                transform: rotate(180deg);
            }}
        </style>
        <script>
            // Table sorting functionality
            function sortTable(tableId, columnIndex, dataType = 'string') {{
                const table = document.getElementById(tableId);
                const tbody = table.getElementsByTagName('tbody')[0];
                const rows = Array.from(tbody.getElementsByTagName('tr'));
                
                // Determine sort direction
                const currentDirection = table.getAttribute('data-sort-direction') || 'asc';
                const newDirection = currentDirection === 'asc' ? 'desc' : 'asc';
                table.setAttribute('data-sort-direction', newDirection);
                
                // Sort rows
                rows.sort((a, b) => {{
                    let aValue = a.getElementsByTagName('td')[columnIndex].textContent.trim();
                    let bValue = b.getElementsByTagName('td')[columnIndex].textContent.trim();
                    
                    if (dataType === 'number') {{
                        // Remove % signs and convert to number
                        aValue = parseFloat(aValue.replace('%', '').replace(',', ''));
                        bValue = parseFloat(bValue.replace('%', '').replace(',', ''));
                        
                        if (newDirection === 'asc') {{
                            return aValue - bValue;
                        }} else {{
                            return bValue - aValue;
                        }}
                    }} else {{
                        if (newDirection === 'asc') {{
                            return aValue.localeCompare(bValue);
                        }} else {{
                            return bValue.localeCompare(aValue);
                        }}
                    }}
                }});
                
                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));
                
                // Update row rankings for summary table
                if (tableId === 'summary-table') {{
                    rows.forEach((row, index) => {{
                        row.getElementsByTagName('td')[0].textContent = index + 1;
                    }});
                }}
                
                // Update header indicators
                const headers = table.getElementsByTagName('th');
                for (let i = 0; i < headers.length; i++) {{
                    if (i === columnIndex) {{
                        headers[i].innerHTML = headers[i].innerHTML.replace(/ [↑↓↕️]/g, '') + (newDirection === 'asc' ? ' ↑' : ' ↓');
                    }} else {{
                        headers[i].innerHTML = headers[i].innerHTML.replace(/ [↑↓↕️]/g, '') + ' ↕️';
                    }}
                }}
            }}
            
            // Collapsible functionality
            function toggleCollapsible(element) {{
                element.classList.toggle('active');
                const content = element.nextElementSibling;
                
                if (element.classList.contains('active')) {{
                    // Expand
                    content.style.maxHeight = content.scrollHeight + 'px';
                    content.classList.add('show');
                }} else {{
                    // Collapse
                    content.style.maxHeight = '0px';
                    content.classList.remove('show');
                }}
            }}
            
            // Initialize collapsibles when page loads
            document.addEventListener('DOMContentLoaded', function() {{
                const collapsibles = document.querySelectorAll('.collapsible');
                collapsibles.forEach(function(collapsible, index) {{
                    const content = collapsible.nextElementSibling;
                    
                    // Only expand the first section (Performance Summary) by default
                    if (index === 0) {{
                        collapsible.classList.add('active');
                        content.style.maxHeight = content.scrollHeight + 'px';
                        content.classList.add('show');
                    }} else {{
                        // Ensure other sections are collapsed
                        content.style.maxHeight = '0px';
                        content.classList.remove('show');
                        collapsible.classList.remove('active');
                    }}
                }});
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Stock Analysis Report</h1>
                <p>Correlation & Return Analysis | Generated: {current_time}</p>
                <p><strong>Stocks:</strong> {', '.join(symbols)} | <strong>Data Period:</strong> {data_period.upper()} | <strong>Time Frame:</strong> {time_frame_days} days</p>
            </div>

            <button class="collapsible" onclick="toggleCollapsible(this)">
                📈 Performance Summary
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
                <table class="summary-table" id="summary-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable('summary-table', 0, 'number')">Rank</th>
                            <th onclick="sortTable('summary-table', 1, 'string')">Symbol</th>
                            <th onclick="sortTable('summary-table', 2, 'number')">Mean Return (%)</th>
                            <th onclick="sortTable('summary-table', 3, 'number')">Success Rate (%)</th>
                            <th onclick="sortTable('summary-table', 4, 'number')">Volatility (%)</th>
                            <th onclick="sortTable('summary-table', 5, 'number')">Sharpe Ratio</th>
                            <th onclick="sortTable('summary-table', 6, 'string')">Assessment</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for rank, stats in enumerate(summary_stats, 1):
        return_class = "positive" if stats['mean_return'] > 0 else "negative" if stats['mean_return'] < 0 else "neutral"
        
        # Risk assessment
        vol = stats['volatility']
        if vol > 15:
            risk_assessment = "🔥 Very High Risk"
        elif vol > 10:
            risk_assessment = "⚡ High Risk"
        elif vol > 5:
            risk_assessment = "📊 Moderate Risk"
        else:
            risk_assessment = "🟢 Low Risk"
        
        html_content += f"""
                        <tr>
                            <td>{rank}</td>
                            <td><strong>{stats['symbol']}</strong></td>
                            <td class="{return_class}">{stats['mean_return']:.2f}%</td>
                            <td>{stats['positive_rate']:.1f}%</td>
                            <td>{stats['volatility']:.2f}%</td>
                            <td>{stats['sharpe_ratio']:.3f}</td>
                            <td>{risk_assessment}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
                
                <div class="info-box">
                    <h4>📊 Analysis Parameters</h4>
                    <p><strong>Time Frame:</strong> {time_frame_days} days | <strong>Data Period:</strong> {data_period.upper()} | <strong>Analysis Date:</strong> {current_time}</p>
                </div>
            </div>

            <button class="collapsible" onclick="toggleCollapsible(this)">
                🔗 Correlation Matrix
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
    """
    
    # Create custom correlation table with sorting
    html_content += '<table class="correlation-table" id="correlation-table"><thead><tr><th onclick="sortTable(\'correlation-table\', 0, \'string\')">Stock</th>'
    
    # Add column headers with sorting
    for i, symbol in enumerate(correlation_matrix.columns):
        html_content += f'<th onclick="sortTable(\'correlation-table\', {i+1}, \'number\')">{symbol}</th>'
    
    html_content += '</tr></thead><tbody>'
    
    # Add correlation data rows
    for index, row in correlation_matrix.iterrows():
        html_content += f'<tr><td><strong>{index}</strong></td>'
        for value in row:
            if pd.isna(value):
                html_content += '<td>-</td>'
            else:
                # Color code correlation values
                if abs(value) > 0.7:
                    color_class = 'style="background-color: #e74c3c; color: white;"' if value > 0 else 'style="background-color: #3498db; color: white;"'
                elif abs(value) > 0.3:
                    color_class = 'style="background-color: #f39c12; color: white;"' if value > 0 else 'style="background-color: #9b59b6; color: white;"'
                else:
                    color_class = 'style="background-color: #95a5a6; color: white;"'
                html_content += f'<td {color_class}>{value:.3f}</td>'
        html_content += '</tr>'
    
    html_content += '</tbody></table></div>'
    
    html_content += f"""
            <button class="collapsible" onclick="toggleCollapsible(this)">
                � Return Distribution Analysis
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
                <div class="chart-container">
                    <img src="data:image/png;base64,{distribution_plots_b64}" alt="Return Distribution Plots">
                </div>
            </div>

            <button class="collapsible" onclick="toggleCollapsible(this)">
                📊 Return Distribution Analysis
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
                <div class="chart-container">
                    <img src="data:image/png;base64,{distribution_plots_b64}" alt="Return Distribution Plots">
                </div>
            </div>

            <button class="collapsible" onclick="toggleCollapsible(this)">
                📊 Return Comparison
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
                <div class="chart-container">
                    <img src="data:image/png;base64,{comparison_chart_b64}" alt="Return Comparison Charts">
                </div>
            </div>

            <button class="collapsible" onclick="toggleCollapsible(this)">
                💡 Key Insights
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
                <div class="info-box">
                    <h4>🎯 How to Read This Analysis</h4>
                    <p><strong>Correlation Matrix:</strong> High correlations (>0.7) suggest stocks move together, limiting diversification benefits.</p>
                    <p><strong>Mean Return:</strong> Average return over {time_frame_days}-day periods. Higher is generally better, but consider volatility.</p>
                    <p><strong>Success Rate:</strong> Percentage of periods with positive returns. Higher rates suggest more consistent performance.</p>
                    <p><strong>Volatility:</strong> Measure of price fluctuation. Higher volatility means more risk but potentially higher returns.</p>
                    <p><strong>Sharpe Ratio:</strong> Risk-adjusted return measure. Higher values indicate better risk-adjusted performance.</p>
                </div>
            </div>

            <button class="collapsible" onclick="toggleCollapsible(this)">
                📊 Detailed Stock Statistics
                <span class="collapsible-icon">▼</span>
            </button>
            <div class="collapsible-content">
                <p style="margin: 5px 0;">Comprehensive analysis for each stock over {time_frame_days}-day periods using {data_period.upper()} of historical data.</p>
    """
    
    # Add detailed statistics for each stock
    for symbol, stats in return_stats_text.items():
        # Risk assessment
        vol = stats['volatility']
        if vol > 15:
            risk_level = "🔥 VERY HIGH"
            risk_color = "#e74c3c"
        elif vol > 10:
            risk_level = "⚡ HIGH"
            risk_color = "#f39c12"
        elif vol > 5:
            risk_level = "📊 MODERATE"
            risk_color = "#3498db"
        else:
            risk_level = "🟢 LOW"
            risk_color = "#27ae60"
        
        html_content += f"""
                <div style="margin: 15px 0; padding: 15px; background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); border-radius: 10px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); border-left: 3px solid {risk_color};">
                    <h3 style="color: #ffffff; margin-top: 0; font-size: 1.2em;">📈 {symbol} - {time_frame_days}-Day Analysis</h3>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin: 10px 0;">
                        
                        <div style="background: rgba(52, 152, 219, 0.15); backdrop-filter: blur(10px); padding: 10px; border-radius: 8px; border: 1px solid rgba(52, 152, 219, 0.3);">
                            <h4 style="color: #74b9ff; margin-top: 0; margin-bottom: 8px;">📈 RETURNS</h4>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Mean:</strong> <span class="{'positive' if stats['mean_return'] > 0 else 'negative' if stats['mean_return'] < 0 else 'neutral'}">{stats['mean_return']:.2f}%</span></p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Median:</strong> {stats['median_return']:.2f}%</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Std Dev:</strong> {stats['std_return']:.2f}%</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Periods:</strong> {stats['total_periods']}</p>
                        </div>
                        
                        <div style="background: rgba(46, 204, 113, 0.15); backdrop-filter: blur(10px); padding: 10px; border-radius: 8px; border: 1px solid rgba(46, 204, 113, 0.3);">
                            <h4 style="color: #00b894; margin-top: 0; margin-bottom: 8px;">✅ POSITIVE</h4>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Count:</strong> {stats['positive_periods']} ({stats['positive_rate']:.1f}%)</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Mean:</strong> <span class="positive">{stats['mean_positive_return']:.2f}%</span></p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Median:</strong> {stats['median_positive_return']:.2f}%</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Max:</strong> <span class="positive">{stats['max_positive_return']:.2f}%</span></p>
                        </div>
                        
                        <div style="background: rgba(231, 76, 60, 0.15); backdrop-filter: blur(10px); padding: 10px; border-radius: 8px; border: 1px solid rgba(231, 76, 60, 0.3);">
                            <h4 style="color: #ff7675; margin-top: 0; margin-bottom: 8px;">❌ NEGATIVE</h4>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Count:</strong> {stats['negative_periods']} ({stats['negative_rate']:.1f}%)</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Mean:</strong> <span class="negative">{stats['mean_negative_return']:.2f}%</span></p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Median:</strong> {stats['median_negative_return']:.2f}%</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Min:</strong> <span class="negative">{stats['max_negative_return']:.2f}%</span></p>
                        </div>
                        
                        <div style="background: rgba(108, 117, 125, 0.15); backdrop-filter: blur(10px); padding: 10px; border-radius: 8px; border: 1px solid rgba(108, 117, 125, 0.3);">
                            <h4 style="color: #a29bfe; margin-top: 0; margin-bottom: 8px;">📊 RISK</h4>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Volatility:</strong> {stats['volatility']:.2f}%</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Sharpe:</strong> {stats['sharpe_ratio']:.3f}</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Sortino:</strong> {stats['sortino_ratio']:.3f}</p>
                            <p style="margin: 3px 0; color: #e8e9ea;"><strong>Risk:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span></p>
                        </div>
                        
                    </div>
                </div>
        """
    
    html_content += """
            </div>

            <div class="footer">
                <p>📊 Generated by Advanced Stock Analysis Tool | {current_time}</p>
                <p>⚠️ This analysis is for informational purposes only and should not be considered as investment advice.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def save_html_report(html_content, filename="index.html"):
    """Save HTML report to file"""
    try:
        file_path = Path(filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📄 HTML report saved as: {file_path.absolute()}")
        print(f"🌐 Open this file in your web browser to view the complete analysis")
        return str(file_path.absolute())
    except Exception as e:
        print(f"❌ Error saving HTML report: {e}")
        return None

def auto_git_commit_and_push():
    """Automatically commit and push changes to git if enabled"""
    try:
        config = load_env_file()
        auto_git_push = config.get('AUTO_GIT_PUSH', 'false').lower() == 'true'
        
        if not auto_git_push:
            return
        
        print("📦 Auto-committing to git...")
        
        # Add all files
        subprocess.run(['git', 'add', '.'], check=True, cwd=Path.cwd())
        
        # Create commit with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_message = config.get('GIT_COMMIT_MESSAGE', 'Auto-update: Stock analysis - {timestamp}')
        commit_message = commit_message.format(timestamp=timestamp)
        
        # Commit changes
        result = subprocess.run(['git', 'commit', '-m', commit_message], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ Git commit created successfully")
            
            # Push to remote
            push_result = subprocess.run(['git', 'push', 'origin', 'main'], 
                                       capture_output=True, text=True, cwd=Path.cwd())
            
            if push_result.returncode == 0:
                print("🚀 Changes pushed to git repository")
            else:
                print(f"⚠️  Push failed: {push_result.stderr}")
                
        elif "nothing to commit" in result.stdout:
            print("ℹ️  No changes to commit")
        else:
            print(f"⚠️  Commit failed: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  Git operation failed: {e}")

def run_analysis_once():
    """Run the analysis once"""
    """Main execution function"""
    try:
        # Load configuration from .env file
        config = load_env_file()
        
        # Get configuration parameters
        data_period = config.get('DATA_PERIOD', '3y')
        time_frame_days = int(config.get('LOOKBACK_DAYS', '28'))
        stock_symbols = config.get('STOCK_SYMBOLS', 'AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX,ORCL,CRM')
        verbose_output = config.get('VERBOSE_OUTPUT', 'false').lower() == 'true'
        
        # Parse stock symbols
        symbols = [s.strip().upper() for s in stock_symbols.split(',') if s.strip()]
        
        if not symbols:
            print("❌ No stock symbols found in .env file")
            print("   Please set STOCK_SYMBOLS in format: TICKER1,TICKER2,TICKER3")
            return
        
        print(f"📋 Configuration loaded from .env:")
        print(f"   • Data Period: {data_period}")
        print(f"   • Lookback Days: {time_frame_days}")
        print(f"   • Stock Symbols: {', '.join(symbols)}")
        print(f"   • Total stocks: {len(symbols)}")
        print(f"   • Verbose Output: {'Enabled' if verbose_output else 'Disabled'}")
        
        if len(symbols) < 2:
            print("❌ Need at least 2 stock symbols for correlation analysis.")
            return
        
        print(f"\n{'='*80}")
        print("STARTING ADVANCED CORRELATION & RETURN ANALYSIS")
        print(f"Data Period: {data_period.upper()} | Time Frame: {time_frame_days} days")
        print(f"{'='*80}")
        
        # Fetch stock data
        price_data, failed_symbols = get_stock_data(symbols, data_period)
        
        if price_data is None:
            print("❌ No data available for analysis.")
            return
        
        # Update symbols list to exclude failed ones
        successful_symbols = [s for s in symbols if s not in failed_symbols]
        
        if len(successful_symbols) < 2:
            print("❌ Need at least 2 stocks with valid data for analysis.")
            return
        
        # Calculate period returns
        period_returns = calculate_period_returns(price_data, time_frame_days)
        
        if period_returns.empty:
            print("❌ Could not calculate period returns.")
            return
        
        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(period_returns)
        
        # Calculate return statistics
        return_stats = calculate_return_statistics(period_returns, successful_symbols, time_frame_days)
        
        if not return_stats:
            print("❌ Could not calculate return statistics.")
            return
        
        # Print analyses (only if verbose output is enabled)
        if verbose_output:
            print_correlation_analysis(correlation_matrix, successful_symbols)
            print_return_statistics_analysis(return_stats, successful_symbols, time_frame_days, data_period)
        
        # Create visualizations and get base64 images for HTML
        print(f"📊 Generating visualizations...")
        distribution_plots_b64 = create_return_distribution_plots(period_returns, successful_symbols, time_frame_days)
        comparison_chart_b64 = create_return_comparison_chart(return_stats, successful_symbols, time_frame_days)
        
        # Generate and save HTML report
        print(f"📄 Generating HTML report...")
        html_content = generate_html_report(
            correlation_matrix, return_stats, successful_symbols, 
            data_period, time_frame_days,
            distribution_plots_b64, comparison_chart_b64
        )
        
        # Get HTML filename from config
        html_filename = config.get('HTML_FILENAME', 'advanced_correlation_analysis.html')
        html_file_path = save_html_report(html_content, html_filename)
        
        if verbose_output:
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE!")
            print("📊 Correlation matrix shows how stocks move together")
            print("📈 Return statistics show performance patterns over specified periods")
            print("💡 Use this analysis for portfolio diversification and risk assessment")
            if html_file_path:
                print(f"🌐 Complete analysis available in HTML format: {html_file_path}")
            print(f"{'='*80}")
        else:
            print("✅ Analysis complete! HTML report generated.")
        
        # Auto commit and push to git if enabled
        auto_git_commit_and_push()
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your inputs and try again.")

def run_scheduler():
    """Run the analysis on a schedule"""
    config = load_env_file()
    interval_hours = int(config.get('SCHEDULE_INTERVAL_HOURS', '4'))
    
    print(f"🕐 Starting automatic stock analysis scheduler")
    print(f"📅 Analysis will run every {interval_hours} hours")
    print(f"⏰ Next run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🛑 Press Ctrl+C to stop the scheduler")
    print("=" * 60)
    
    try:
        while True:
            # Run analysis
            print(f"\n🚀 Starting scheduled analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

def main():
    """Main function - can run once or start scheduler"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['schedule', 'scheduler', '--schedule', '-s']:
            run_scheduler()
        elif sys.argv[1].lower() in ['once', '--once', '-o']:
            run_analysis_once()
        elif sys.argv[1].lower() in ['help', '--help', '-h']:
            print("Stock Correlation Analysis Tool")
            print("=" * 40)
            print("Usage:")
            print("  python3 advanced_correlation_analysis.py          # Run once")
            print("  python3 advanced_correlation_analysis.py once     # Run once")
            print("  python3 advanced_correlation_analysis.py schedule # Run every 4 hours")
            print("  python3 advanced_correlation_analysis.py help     # Show this help")
            print("\nConfiguration:")
            print("  Edit .env file to change settings:")
            print("  - SCHEDULE_INTERVAL_HOURS: How often to run (default: 4)")
            print("  - AUTO_GIT_PUSH: Automatically push to git (default: false)")
            print("  - VERBOSE_OUTPUT: Show detailed output (default: false)")
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use 'help' to see available options")
    else:
        # Default behavior - run once
        run_analysis_once()

if __name__ == "__main__":
    main()
