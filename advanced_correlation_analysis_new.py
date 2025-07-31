import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import base64
import io
import json
import shutil
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
            
            if data.empty or len(data) < 50:
                print(f"    ❌ {symbol}: Insufficient data")
                failed_symbols.append(symbol)
            else:
                stock_data[symbol] = data['Close']
                print(f"    ✅ {symbol}: {len(data)} trading days")
                
        except Exception as e:
            print(f"    ❌ {symbol}: Failed to download - {str(e)}")
            failed_symbols.append(symbol)
            continue
    
    if not stock_data:
        raise ValueError("No valid stock data retrieved. Please check your symbols.")
    
    # Create DataFrame with closing prices
    df = pd.DataFrame(stock_data)
    df = df.dropna()
    
    return df, failed_symbols

def calculate_period_returns(stock_data, lookback_days=30):
    """
    Calculate returns for specific time periods (e.g., monthly returns)
    
    Parameters:
    - stock_data: DataFrame with stock prices
    - lookback_days: Number of days per period
    
    Returns:
    - DataFrame with period returns
    """
    print(f"📈 Calculating {lookback_days}-day period returns...")
    
    returns_data = {}
    
    for symbol in stock_data.columns:
        prices = stock_data[symbol].dropna()
        
        if len(prices) < lookback_days * 2:
            print(f"  ⚠️ {symbol}: Insufficient data for {lookback_days}-day periods")
            continue
        
        period_returns = []
        
        # Calculate returns for each period
        for i in range(lookback_days, len(prices), lookback_days):
            start_price = prices.iloc[i - lookback_days]
            end_price = prices.iloc[i]
            
            if start_price > 0:  # Avoid division by zero
                period_return = ((end_price - start_price) / start_price) * 100
                period_returns.append(period_return)
        
        if period_returns:
            returns_data[symbol] = period_returns
    
    # Create DataFrame with equal length series (pad with NaN if needed)
    max_length = max(len(returns) for returns in returns_data.values()) if returns_data else 0
    
    for symbol in returns_data:
        current_length = len(returns_data[symbol])
        if current_length < max_length:
            returns_data[symbol].extend([np.nan] * (max_length - current_length))
    
    period_returns_df = pd.DataFrame(returns_data)
    return period_returns_df

def save_chart_to_base64(fig):
    """Save matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close(fig)
    return image_base64

def create_correlation_heatmap(correlation_matrix, title):
    """Create correlation heatmap and return base64 image"""
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='white')
    plt.tight_layout()
    
    return save_chart_to_base64(fig)

def create_return_distribution_plots(period_returns, symbols, time_frame_days):
    """Create return distribution plots and return base64 image"""
    plt.style.use('dark_background')
    
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
            axes[i].text(0.5, 0.5, f'No data\\nfor {symbol}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(symbol, color='white')
            continue
        
        # Create histogram
        axes[i].hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {returns.mean():.1f}%')
        axes[i].axvline(returns.median(), color='green', linestyle='--', 
                       label=f'Median: {returns.median():.1f}%')
        
        axes[i].set_title(f'{symbol}', fontweight='bold', color='white')
        axes[i].set_xlabel('Return (%)', color='white')
        axes[i].set_ylabel('Frequency', color='white')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Color tick labels
        axes[i].tick_params(colors='white')
    
    # Hide empty subplots
    for i in range(len(symbols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Return Distribution Analysis ({time_frame_days}-Day Periods)', 
                 fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    
    return save_chart_to_base64(fig)

def create_return_comparison_chart(return_stats, symbols, time_frame_days):
    """Create comprehensive return comparison charts and return base64 image"""
    plt.style.use('dark_background')
    
    # Extract data for plotting
    mean_returns = [return_stats[symbol]['mean_return'] for symbol in symbols if symbol in return_stats]
    positive_rates = [return_stats[symbol]['positive_rate'] for symbol in symbols if symbol in return_stats]
    volatilities = [return_stats[symbol]['volatility'] for symbol in symbols if symbol in return_stats]
    max_gains = [return_stats[symbol]['max_positive_return'] for symbol in symbols if symbol in return_stats]
    max_losses = [return_stats[symbol]['max_negative_return'] for symbol in symbols if symbol in return_stats]
    valid_symbols = [symbol for symbol in symbols if symbol in return_stats]
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Mean Returns
    bars1 = ax1.bar(valid_symbols, mean_returns, 
                    color=['green' if x > 0 else 'red' for x in mean_returns])
    ax1.set_title('Mean Returns by Stock', fontweight='bold', color='white')
    ax1.set_ylabel('Mean Return (%)', color='white')
    ax1.tick_params(axis='x', rotation=45, colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='white', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mean_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', color='white')
    
    # Chart 2: Positive Return Rates
    bars2 = ax2.bar(valid_symbols, positive_rates, color='lightblue')
    ax2.set_title('Positive Return Rate by Stock', fontweight='bold', color='white')
    ax2.set_ylabel('Positive Return Rate (%)', color='white')
    ax2.tick_params(axis='x', rotation=45, colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, positive_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', color='white')
    
    # Chart 3: Max Gains vs Max Losses
    x_pos = np.arange(len(valid_symbols))
    width = 0.35
    
    bars3a = ax3.bar(x_pos - width/2, max_gains, width, label='Max Gains', color='green', alpha=0.7)
    bars3b = ax3.bar(x_pos + width/2, [abs(x) for x in max_losses], width, label='Max Losses', color='red', alpha=0.7)
    
    ax3.set_title('Maximum Gains vs Losses', fontweight='bold', color='white')
    ax3.set_ylabel('Return (%)', color='white')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(valid_symbols, rotation=45)
    ax3.tick_params(colors='white')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Volatility
    bars4 = ax4.bar(valid_symbols, volatilities, color='orange')
    ax4.set_title('Volatility by Stock', fontweight='bold', color='white')
    ax4.set_ylabel('Volatility (%)', color='white')
    ax4.tick_params(axis='x', rotation=45, colors='white')
    ax4.tick_params(axis='y', colors='white')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, vol in zip(bars4, volatilities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{vol:.1f}%', ha='center', va='bottom', fontweight='bold', color='white')
    
    plt.suptitle(f'Comprehensive Return Analysis ({time_frame_days}-Day Periods)', 
                 fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    
    return save_chart_to_base64(fig)

def calculate_return_statistics(period_returns, symbols, lookback_days, risk_free_rate=0.02):
    """
    Calculate comprehensive return statistics for each stock
    
    Parameters:
    - period_returns: DataFrame with period returns
    - symbols: List of stock symbols
    - risk_free_rate: Risk-free rate for Sharpe ratio calculation
    
    Returns:
    - Dictionary with statistics for each stock
    """
    print("📊 Calculating return statistics for {}-day periods...".format(len(period_returns)))
    
    stats = {}
    
    for symbol in symbols:
        if symbol not in period_returns.columns:
            continue
            
        returns = period_returns[symbol].dropna()
        
        if len(returns) == 0:
            continue
        
        # Basic statistics
        mean_return = returns.mean()
        median_return = returns.median()
        std_return = returns.std()
        
        # Positive/Negative breakdown
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        positive_count = len(positive_returns)
        negative_count = len(negative_returns)
        total_periods = len(returns)
        
        positive_rate = (positive_count / total_periods) * 100
        negative_rate = (negative_count / total_periods) * 100
        
        # Extreme values
        max_positive = positive_returns.max() if len(positive_returns) > 0 else 0
        max_negative = negative_returns.min() if len(negative_returns) > 0 else 0
        
        mean_positive = positive_returns.mean() if len(positive_returns) > 0 else 0
        mean_negative = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        median_positive = positive_returns.median() if len(positive_returns) > 0 else 0
        median_negative = negative_returns.median() if len(negative_returns) > 0 else 0
        
        # Risk metrics (annualized approximations)
        # Assuming the lookback period represents a fraction of a year
        periods_per_year = 252 / lookback_days  # Approximate
        volatility = std_return * np.sqrt(periods_per_year)
        
        # Sharpe ratio (annualized)
        annualized_return = mean_return * periods_per_year
        sharpe_ratio = (annualized_return - risk_free_rate * 100) / volatility if volatility > 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - risk_free_rate * 100) / downside_deviation if downside_deviation > 0 else 0
        
        stats[symbol] = {
            'symbol': symbol,
            'mean_return': mean_return,
            'median_return': median_return,
            'std_return': std_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_periods': total_periods,
            'positive_periods': positive_count,
            'negative_periods': negative_count,
            'positive_rate': positive_rate,
            'negative_rate': negative_rate,
            'max_positive_return': max_positive,
            'max_negative_return': max_negative,
            'mean_positive_return': mean_positive,
            'mean_negative_return': mean_negative,
            'median_positive_return': median_positive,
            'median_negative_return': median_negative
        }
    
    return stats

def generate_assets(stock_data, period_returns, symbols, data_period, time_frame_days, 
                   correlation_data, return_stats, env_vars):
    """Generate JSON data and chart assets"""
    print("📄 Generating assets...")
    
    # Ensure assets directory exists
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Generate charts
    charts = []
    
    # Return distribution chart
    dist_chart = create_return_distribution_plots(period_returns, symbols, time_frame_days)
    charts.append({
        "title": f"Return Distribution Analysis ({time_frame_days}-Day Periods)",
        "image": dist_chart
    })
    
    # Return comparison chart
    comp_chart = create_return_comparison_chart(return_stats, symbols, time_frame_days)
    charts.append({
        "title": f"Comprehensive Return Analysis ({time_frame_days}-Day Periods)",
        "image": comp_chart
    })
    
    # Correlation heatmap (if enabled)
    if env_vars.get('SHOW_CORRELATION_HEATMAP', 'false').lower() == 'true':
        corr_matrix = stock_data.corr()
        heatmap_chart = create_correlation_heatmap(corr_matrix, f"Stock Correlation Heatmap ({data_period})")
        charts.append({
            "title": f"Stock Correlation Heatmap ({data_period})",
            "image": heatmap_chart
        })
    
    # Prepare summary statistics
    summary_stats = []
    for symbol in symbols:
        if symbol in return_stats:
            stats = return_stats[symbol]
            summary_stats.append(stats)
    
    # Sort by mean return for display
    summary_stats.sort(key=lambda x: x['mean_return'], reverse=True)
    
    # Prepare correlation matrix for JSON
    correlation_matrix = []
    if correlation_data is not None and not correlation_data.empty:
        corr_df = stock_data.corr()
        for i, symbol1 in enumerate(symbols):
            if symbol1 not in corr_df.index:
                continue
            row = []
            for j, symbol2 in enumerate(symbols):
                if symbol2 not in corr_df.columns:
                    continue
                correlation = corr_df.loc[symbol1, symbol2]
                row.append({
                    "symbol": symbol2,
                    "correlation": float(correlation) if not pd.isna(correlation) else 0.0
                })
            if row:
                correlation_matrix.append(row)
    
    # Prepare performance stats (individual stock details)
    performance_stats = []
    for symbol in symbols:
        if symbol in return_stats:
            stats = return_stats[symbol].copy()
            stats['time_frame_days'] = time_frame_days
            performance_stats.append(stats)
    
    # Create comprehensive data object
    data = {
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_period": data_period,
        "time_frame_days": time_frame_days,
        "symbols": symbols,
        "summary_stats": summary_stats,
        "charts": charts,
        "correlation_matrix": correlation_matrix,
        "performance_stats": performance_stats
    }
    
    # Save to JSON file
    json_file = assets_dir / "correlation_data.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"📄 Assets saved to: {assets_dir}")
    return str(json_file)

def copy_template_to_output():
    """Copy the HTML template to the final output location"""
    template_path = Path("templates/advanced_correlation_analysis.html")
    output_path = Path("advanced_correlation_analysis.html")
    
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
        commit_message = f"Auto-update: Stock analysis - {timestamp}"
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
            print("🔄 SCHEDULED ANALYSIS STARTING")
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
    data_period = env_vars.get('DATA_PERIOD', '5y')
    lookback_days = int(env_vars.get('LOOKBACK_DAYS', 30))
    symbols_str = env_vars.get('STOCK_SYMBOLS', 'AAPL,MSFT,GOOGL,AMZN,TSLA')
    risk_free_rate = float(env_vars.get('RISK_FREE_RATE', 0.02))
    min_correlation = float(env_vars.get('MIN_CORRELATION_HIGHLIGHT', 0.7))
    html_filename = env_vars.get('HTML_FILENAME', 'advanced_correlation_analysis.html')
    verbose = env_vars.get('VERBOSE_OUTPUT', 'false').lower() == 'true'
    auto_git_push = env_vars.get('AUTO_GIT_PUSH', 'false').lower() == 'true'
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    
    print("📋 Configuration loaded from .env:")
    print(f"   • Data Period: {data_period}")
    print(f"   • Lookback Days: {lookback_days}")
    print(f"   • Stock Symbols: {', '.join(symbols)}")
    print(f"   • Total stocks: {len(symbols)}")
    print(f"   • Verbose Output: {'Enabled' if verbose else 'Disabled'}")
    
    print("="*80)
    print("STARTING ADVANCED CORRELATION & RETURN ANALYSIS")
    print(f"Data Period: {data_period.upper()} | Time Frame: {lookback_days} days")
    print("="*80)
    
    try:
        # Fetch stock data
        stock_data, failed_symbols = get_stock_data(symbols, period=data_period)
        
        if failed_symbols:
            print(f"\\n⚠️  Failed to fetch data for: {', '.join(failed_symbols)}")
            symbols = [s for s in symbols if s not in failed_symbols]
        
        # Calculate period returns
        period_returns = calculate_period_returns(stock_data, lookback_days)
        
        # Calculate statistics
        return_stats = calculate_return_statistics(period_returns, symbols, lookback_days, risk_free_rate)
        
        # Generate visualizations and prepare data
        print("📊 Generating visualizations...")
        correlation_data = stock_data.corr() if len(symbols) > 1 else None
        
        # Generate assets (JSON data and images)
        json_file = generate_assets(
            stock_data, period_returns, symbols, data_period, 
            lookback_days, correlation_data, return_stats, env_vars
        )
        
        # Copy template to output location
        html_file = copy_template_to_output()
        
        if html_file:
            print(f"📄 HTML report saved as: {os.path.abspath(html_file)}")
            print("🌐 Open this file in your web browser to view the complete analysis")
        
        print("✅ Analysis complete! HTML report generated.")
        
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
        print("📅 Starting scheduled analysis...")
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
