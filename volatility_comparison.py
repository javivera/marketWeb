import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def calculate_volatility_metrics(price_data, symbols):
    """
    Calculate various volatility metrics for each stock
    
    Parameters:
    - price_data: DataFrame with stock prices
    - symbols: List of stock symbols
    
    Returns:
    - Dictionary with volatility metrics
    """
    print(f"\n📈 Calculating volatility metrics...")
    
    volatility_data = {}
    
    for symbol in symbols:
        if symbol not in price_data.columns:
            continue
            
        prices = price_data[symbol].dropna()
        
        if len(prices) < 30:  # Need at least 30 days of data
            print(f"⚠️  {symbol}: Insufficient data ({len(prices)} days)")
            continue
        
        # Calculate daily returns
        daily_returns = prices.pct_change().dropna()
        
        # Calculate various volatility metrics
        metrics = {
            'daily_volatility': daily_returns.std(),
            'annualized_volatility': daily_returns.std() * np.sqrt(252),
            'rolling_30_volatility': daily_returns.rolling(30).std().iloc[-1] * np.sqrt(252),
            'rolling_60_volatility': daily_returns.rolling(60).std().iloc[-1] * np.sqrt(252) if len(daily_returns) >= 60 else np.nan,
            'max_daily_return': daily_returns.max(),
            'min_daily_return': daily_returns.min(),
            'average_return': daily_returns.mean(),
            'annualized_return': daily_returns.mean() * 252,
            'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)),
            'current_price': prices.iloc[-1],
            'price_change_period': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
            'max_drawdown': calculate_max_drawdown(prices),
            'trading_days': len(prices),
            'volatility_rank': 0  # Will be calculated later
        }
        
        volatility_data[symbol] = metrics
    
    # Calculate volatility rankings
    vol_values = [(symbol, data['annualized_volatility']) for symbol, data in volatility_data.items()]
    vol_values.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (symbol, _) in enumerate(vol_values, 1):
        volatility_data[symbol]['volatility_rank'] = rank
    
    return volatility_data

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices / peak - 1.0)
    return drawdown.min()

def print_volatility_analysis(volatility_data, period):
    """Print detailed volatility analysis"""
    
    print(f"\n{'VOLATILITY COMPARISON ANALYSIS':-^80}")
    print(f"Period: {period.upper()} | Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*80}")
    
    # Sort by annualized volatility (highest to lowest)
    sorted_stocks = sorted(volatility_data.items(), 
                          key=lambda x: x[1]['annualized_volatility'], 
                          reverse=True)
    
    print(f"\n{'VOLATILITY RANKINGS (Highest to Lowest)':-^80}")
    print(f"{'Rank':<5} {'Symbol':<8} {'Ann. Vol':<10} {'Daily Vol':<10} {'Sharpe':<8} {'Return':<10}")
    print("-" * 80)
    
    for symbol, data in sorted_stocks:
        rank = data['volatility_rank']
        ann_vol = data['annualized_volatility']
        daily_vol = data['daily_volatility']
        sharpe = data['sharpe_ratio']
        ann_return = data['annualized_return']
        
        print(f"{rank:<5} {symbol:<8} {ann_vol:<10.2%} {daily_vol:<10.2%} {sharpe:<8.2f} {ann_return:<10.2%}")
    
    print(f"\n{'DETAILED STATISTICS':-^80}")
    
    for symbol, data in sorted_stocks:
        print(f"\n📊 {symbol} - Volatility Analysis")
        print(f"  • Current Price: ${data['current_price']:.2f}")
        print(f"  • Period Return: {data['price_change_period']:+.1f}%")
        print(f"  • Annualized Volatility: {data['annualized_volatility']:.2%}")
        print(f"  • Daily Volatility: {data['daily_volatility']:.2%}")
        
        if not np.isnan(data['rolling_30_volatility']):
            print(f"  • 30-Day Rolling Volatility: {data['rolling_30_volatility']:.2%}")
        if not np.isnan(data['rolling_60_volatility']):
            print(f"  • 60-Day Rolling Volatility: {data['rolling_60_volatility']:.2%}")
        
        print(f"  • Annualized Return: {data['annualized_return']:.2%}")
        print(f"  • Sharpe Ratio: {data['sharpe_ratio']:.3f}")
        print(f"  • Max Daily Gain: {data['max_daily_return']:.2%}")
        print(f"  • Max Daily Loss: {data['min_daily_return']:.2%}")
        print(f"  • Maximum Drawdown: {data['max_drawdown']:.2%}")
        print(f"  • Trading Days: {data['trading_days']}")
        
        # Volatility assessment
        vol = data['annualized_volatility']
        if vol > 0.40:
            assessment = "🔥 EXTREMELY HIGH"
        elif vol > 0.30:
            assessment = "📈 VERY HIGH"
        elif vol > 0.20:
            assessment = "⚡ HIGH"
        elif vol > 0.15:
            assessment = "📊 MODERATE"
        else:
            assessment = "🟢 LOW"
        
        print(f"  • Risk Assessment: {assessment}")

def create_volatility_comparison_chart(volatility_data):
    """Create comprehensive volatility comparison charts"""
    
    symbols = list(volatility_data.keys())
    ann_vols = [volatility_data[symbol]['annualized_volatility'] for symbol in symbols]
    ann_returns = [volatility_data[symbol]['annualized_return'] for symbol in symbols]
    sharpe_ratios = [volatility_data[symbol]['sharpe_ratio'] for symbol in symbols]
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Volatility Bar Chart
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(symbols)))
    bars = ax1.bar(symbols, [v * 100 for v in ann_vols], color=colors)
    ax1.set_title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Volatility (%)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, vol in zip(bars, ann_vols):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{vol:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Risk-Return Scatter Plot
    scatter = ax2.scatter([v * 100 for v in ann_vols], [r * 100 for r in ann_returns], 
                         s=100, alpha=0.7, c=colors)
    ax2.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax2.set_ylabel('Annualized Return (%)', fontsize=12)
    ax2.set_title('Risk vs Return Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add stock labels to scatter plot
    for i, symbol in enumerate(symbols):
        ax2.annotate(symbol, 
                    (ann_vols[i] * 100, ann_returns[i] * 100),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    # 3. Sharpe Ratio Comparison
    colors_sharpe = ['green' if s > 0 else 'red' for s in sharpe_ratios]
    bars_sharpe = ax3.bar(symbols, sharpe_ratios, color=colors_sharpe, alpha=0.7)
    ax3.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sharpe in zip(bars_sharpe, sharpe_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., 
                height + (0.05 if height >= 0 else -0.15),
                f'{sharpe:.2f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 4. Volatility Heatmap
    vol_matrix = np.array([ann_vols]).T
    im = ax4.imshow(vol_matrix, cmap='RdYlBu_r', aspect='auto')
    ax4.set_title('Volatility Heatmap', fontsize=14, fontweight='bold')
    ax4.set_yticks(range(len(symbols)))
    ax4.set_yticklabels(symbols)
    ax4.set_xticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Annualized Volatility', rotation=270, labelpad=20)
    
    # Add volatility values to heatmap
    for i, vol in enumerate(ann_vols):
        ax4.text(0, i, f'{vol:.1%}', ha='center', va='center', 
                fontweight='bold', color='white' if vol > 0.25 else 'black')
    
    plt.tight_layout()
    print(f"\n📊 Displaying volatility comparison charts...")
    plt.show()

def create_volatility_correlation_heatmap(price_data, symbols):
    """Create correlation heatmap of stock returns"""
    
    # Calculate daily returns for all stocks
    returns_data = price_data.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns_data.corr()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'Correlation'})
    
    plt.title('Stock Returns Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    print(f"\n📊 Displaying correlation heatmap...")
    plt.show()

def get_user_input():
    """Get user input for volatility analysis"""
    print("=== Stock Volatility Comparison Analysis ===\n")
    
    # Get stock symbols
    print("Enter stock symbols (separated by commas or spaces):")
    print("Example: AAPL, GOOGL, MSFT, TSLA, NVDA")
    symbols_input = input("Stock symbols: ").strip()
    
    # Parse symbols
    symbols = []
    for symbol in symbols_input.replace(',', ' ').split():
        symbol = symbol.strip().upper()
        if symbol:
            symbols.append(symbol)
    
    if len(symbols) < 2:
        print("❌ Please enter at least 2 stock symbols.")
        return None, None
    
    print(f"✅ Analyzing {len(symbols)} stocks: {', '.join(symbols)}")
    
    # Get time period
    print("\nSelect time period:")
    print("1. 1 year (recommended)")
    print("2. 2 years")
    print("3. 3 years")
    print("4. 5 years")
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            period = "1y"
            break
        elif choice == "2":
            period = "2y"
            break
        elif choice == "3":
            period = "3y"
            break
        elif choice == "4":
            period = "5y"
            break
        else:
            print("Please enter 1, 2, 3, or 4.")
    
    return symbols, period

def main():
    """Main execution function"""
    try:
        # Get user input
        symbols, period = get_user_input()
        
        if symbols is None:
            return
        
        print(f"\n{'='*80}")
        print("STARTING VOLATILITY ANALYSIS")
        print(f"{'='*80}")
        
        # Fetch stock data
        price_data, failed_symbols = get_stock_data(symbols, period)
        
        if price_data is None:
            print("❌ No data available for analysis.")
            return
        
        # Update symbols list to exclude failed ones
        successful_symbols = [s for s in symbols if s not in failed_symbols]
        
        if len(successful_symbols) < 2:
            print("❌ Need at least 2 stocks with valid data for comparison.")
            return
        
        # Calculate volatility metrics
        volatility_data = calculate_volatility_metrics(price_data, successful_symbols)
        
        if not volatility_data:
            print("❌ Could not calculate volatility metrics.")
            return
        
        # Print analysis
        print_volatility_analysis(volatility_data, period)
        
        # Create visualizations
        create_volatility_comparison_chart(volatility_data)
        create_volatility_correlation_heatmap(price_data, successful_symbols)
        
        print(f"\n{'='*80}")
        print("VOLATILITY ANALYSIS COMPLETE!")
        print("💡 Higher volatility = Higher risk but potentially higher returns")
        print("📊 Sharpe ratio shows risk-adjusted returns (higher is better)")
        print("🔗 Correlation shows how stocks move together")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
