import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def get_stock_data(symbols, period='1y'):
    """
    Download stock data for multiple symbols
    
    Parameters:
    - symbols (list): List of stock symbols
    - period (str): Time period for data ('1y', '2y', '3y', '5y')
    
    Returns:
    - DataFrame: Combined closing prices for all stocks
    """
    print(f"Downloading data for {len(symbols)} stocks over {period} period...")
    
    all_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            print(f"  Fetching {symbol}...")
            data = yf.download(symbol, period=period, interval='1d', progress=False)
            
            if data.empty:
                print(f"  ❌ No data found for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Store closing prices
            all_data[symbol] = data['Close']
            print(f"  ✅ {symbol}: {len(data)} data points")
            
        except Exception as e:
            print(f"  ❌ Error downloading {symbol}: {e}")
            failed_symbols.append(symbol)
    
    if not all_data:
        print("❌ No valid data found for any symbols!")
        return None, failed_symbols
    
    # Combine all data into one DataFrame
    combined_data = pd.DataFrame(all_data)
    
    # Remove rows with any NaN values
    initial_rows = len(combined_data)
    combined_data = combined_data.dropna()
    final_rows = len(combined_data)
    
    if final_rows < initial_rows:
        print(f"⚠️  Removed {initial_rows - final_rows} rows with missing data")
    
    print(f"✅ Successfully loaded data for {len(all_data)} stocks")
    print(f"   Date range: {combined_data.index[0].strftime('%Y-%m-%d')} to {combined_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Total data points: {len(combined_data)}")
    
    if failed_symbols:
        print(f"⚠️  Failed to load: {', '.join(failed_symbols)}")
    
    return combined_data, failed_symbols

def calculate_returns(price_data, return_type='daily'):
    """
    Calculate returns from price data
    
    Parameters:
    - price_data (DataFrame): Stock price data
    - return_type (str): 'daily', 'weekly', 'monthly'
    
    Returns:
    - DataFrame: Returns data
    """
    if return_type == 'daily':
        returns = price_data.pct_change().dropna()
    elif return_type == 'weekly':
        weekly_prices = price_data.resample('W').last()
        returns = weekly_prices.pct_change().dropna()
    elif return_type == 'monthly':
        monthly_prices = price_data.resample('M').last()
        returns = monthly_prices.pct_change().dropna()
    else:
        raise ValueError("return_type must be 'daily', 'weekly', or 'monthly'")
    
    # Convert to percentage
    returns = returns * 100
    
    print(f"✅ Calculated {return_type} returns: {len(returns)} observations")
    return returns

def calculate_correlation_matrix(returns_data):
    """Calculate correlation matrix"""
    correlation_matrix = returns_data.corr()
    return correlation_matrix

def calculate_covariance_matrix(returns_data):
    """Calculate covariance matrix"""
    covariance_matrix = returns_data.cov()
    return covariance_matrix

def print_correlation_analysis(correlation_matrix, symbols):
    """Print correlation analysis results"""
    print(f"\n{'CORRELATION MATRIX':-^80}")
    print("Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)")
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    # Find highest and lowest correlations (excluding diagonal)
    print(f"\n{'CORRELATION INSIGHTS':-^80}")
    
    # Create a copy and set diagonal to NaN to exclude self-correlations
    corr_no_diag = correlation_matrix.copy()
    np.fill_diagonal(corr_no_diag.values, np.nan)
    
    # Find highest correlation
    max_corr = corr_no_diag.max().max()
    max_corr_idx = corr_no_diag.stack().idxmax()
    
    # Find lowest correlation
    min_corr = corr_no_diag.min().min()
    min_corr_idx = corr_no_diag.stack().idxmin()
    
    print(f"Highest correlation: {max_corr_idx[0]} & {max_corr_idx[1]} = {max_corr:.3f}")
    print(f"Lowest correlation: {min_corr_idx[0]} & {min_corr_idx[1]} = {min_corr:.3f}")
    
    # Average correlation
    avg_corr = corr_no_diag.mean().mean()
    print(f"Average correlation (excluding self): {avg_corr:.3f}")
    
    # Correlation categories
    print(f"\n{'CORRELATION STRENGTH ANALYSIS':-^80}")
    strong_positive = (corr_no_diag > 0.7).sum().sum()
    moderate_positive = ((corr_no_diag > 0.3) & (corr_no_diag <= 0.7)).sum().sum()
    weak_correlation = ((corr_no_diag >= -0.3) & (corr_no_diag <= 0.3)).sum().sum()
    moderate_negative = ((corr_no_diag >= -0.7) & (corr_no_diag < -0.3)).sum().sum()
    strong_negative = (corr_no_diag < -0.7).sum().sum()
    
    total_pairs = len(symbols) * (len(symbols) - 1)  # Exclude diagonal, count both directions
    
    print(f"Strong positive (>0.7): {strong_positive} pairs ({strong_positive/total_pairs*100:.1f}%)")
    print(f"Moderate positive (0.3-0.7): {moderate_positive} pairs ({moderate_positive/total_pairs*100:.1f}%)")
    print(f"Weak correlation (-0.3 to 0.3): {weak_correlation} pairs ({weak_correlation/total_pairs*100:.1f}%)")
    print(f"Moderate negative (-0.7 to -0.3): {moderate_negative} pairs ({moderate_negative/total_pairs*100:.1f}%)")
    print(f"Strong negative (<-0.7): {strong_negative} pairs ({strong_negative/total_pairs*100:.1f}%)")

def print_covariance_analysis(covariance_matrix, symbols):
    """Print covariance analysis results"""
    print(f"\n{'COVARIANCE MATRIX':-^80}")
    print("Covariance measures how returns move together (in percentage points squared)")
    print("\nCovariance Matrix:")
    print(covariance_matrix.round(4))
    
    print(f"\n{'COVARIANCE INSIGHTS':-^80}")
    
    # Diagonal elements are variances
    variances = np.diag(covariance_matrix)
    volatilities = np.sqrt(variances)
    
    print("Individual Stock Volatilities (Standard Deviation):")
    for i, symbol in enumerate(symbols):
        print(f"  {symbol}: {volatilities[i]:.2f}%")
    
    print(f"\nAverage volatility: {volatilities.mean():.2f}%")
    print(f"Highest volatility: {symbols[np.argmax(volatilities)]} ({volatilities.max():.2f}%)")
    print(f"Lowest volatility: {symbols[np.argmin(volatilities)]} ({volatilities.min():.2f}%)")

def create_correlation_heatmap(correlation_matrix, title="Stock Correlation Matrix"):
    """Create a correlation heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Stocks', fontsize=12)
    plt.ylabel('Stocks', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    print(f"\n📊 Displaying correlation heatmap...")
    plt.show()

def create_covariance_heatmap(covariance_matrix, title="Stock Covariance Matrix"):
    """Create a covariance heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(covariance_matrix, 
                annot=True, 
                cmap='viridis', 
                square=True,
                fmt='.4f',
                cbar_kws={'label': 'Covariance'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Stocks', fontsize=12)
    plt.ylabel('Stocks', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    print(f"\n📊 Displaying covariance heatmap...")
    plt.show()

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate negative Sharpe ratio for optimization (minimize negative = maximize positive)"""
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Calculate portfolio volatility for minimum variance optimization"""
    _, portfolio_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return portfolio_vol

def optimize_portfolio(mean_returns, cov_matrix, optimization_type='sharpe'):
    """
    Optimize portfolio weights
    
    Parameters:
    - mean_returns: Expected returns for each asset
    - cov_matrix: Covariance matrix
    - optimization_type: 'sharpe' for max Sharpe ratio, 'min_vol' for minimum variance
    
    Returns:
    - Dictionary with optimization results
    """
    num_assets = len(mean_returns)
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling, max 100% per asset
    
    # Initial guess (equal weights)
    initial_guess = np.array([1/num_assets] * num_assets)
    
    if optimization_type == 'sharpe':
        # Maximize Sharpe ratio
        result = minimize(negative_sharpe_ratio, initial_guess,
                         args=(mean_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
            
            return {
                'type': 'Maximum Sharpe Ratio',
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
    
    elif optimization_type == 'min_vol':
        # Minimize volatility
        result = minimize(portfolio_volatility, initial_guess,
                         args=(mean_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol
            
            return {
                'type': 'Minimum Variance',
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'success': True
            }
    
    return {'success': False, 'message': 'Optimization failed'}

def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    """Generate efficient frontier"""
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    
    # Calculate individual asset performance for bounds
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    for i, target in enumerate(target_returns):
        # Add return target constraint
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) * 252 - target},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(portfolio_volatility, np.array([1/num_assets] * num_assets),
                         args=(mean_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            portfolio_return, portfolio_vol = portfolio_performance(result.x, mean_returns, cov_matrix)
            results[0, i] = target
            results[1, i] = portfolio_vol
            results[2, i] = (target - 0.02) / portfolio_vol  # Sharpe ratio
    
    return results

def print_portfolio_optimization(returns_data, symbols):
    """Perform and display portfolio optimization"""
    print(f"\n{'PORTFOLIO OPTIMIZATION':-^80}")
    
    # Calculate mean returns and covariance matrix (daily)
    mean_returns = returns_data.mean() / 100  # Convert from percentage
    cov_matrix = returns_data.cov() / 10000   # Convert from percentage squared
    
    print("Expected Annual Returns (based on historical average):")
    annual_returns = mean_returns * 252
    for i, symbol in enumerate(symbols):
        print(f"  {symbol}: {annual_returns[i]:.2%}")
    
    print("\nAnnual Volatilities:")
    annual_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
    for i, symbol in enumerate(symbols):
        print(f"  {symbol}: {annual_vols[i]:.2%}")
    
    # Equal weight portfolio (benchmark)
    equal_weights = np.array([1/len(symbols)] * len(symbols))
    equal_return, equal_vol = portfolio_performance(equal_weights, mean_returns, cov_matrix)
    equal_sharpe = (equal_return - 0.02) / equal_vol
    
    print(f"\n{'EQUAL WEIGHT PORTFOLIO (BENCHMARK)':-^80}")
    print(f"Expected Return: {equal_return:.2%}")
    print(f"Volatility: {equal_vol:.2%}")
    print(f"Sharpe Ratio: {equal_sharpe:.3f}")
    print("Weights:")
    for i, symbol in enumerate(symbols):
        print(f"  {symbol}: {equal_weights[i]:.1%}")
    
    # Optimize for maximum Sharpe ratio
    max_sharpe_result = optimize_portfolio(mean_returns, cov_matrix, 'sharpe')
    
    if max_sharpe_result['success']:
        print(f"\n{max_sharpe_result['type'].upper() + ' PORTFOLIO':-^80}")
        print(f"Expected Return: {max_sharpe_result['expected_return']:.2%}")
        print(f"Volatility: {max_sharpe_result['volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.3f}")
        print("Optimal Weights:")
        for i, symbol in enumerate(symbols):
            print(f"  {symbol}: {max_sharpe_result['weights'][i]:.1%}")
    
    # Optimize for minimum variance
    min_var_result = optimize_portfolio(mean_returns, cov_matrix, 'min_vol')
    
    if min_var_result['success']:
        print(f"\n{min_var_result['type'].upper() + ' PORTFOLIO':-^80}")
        print(f"Expected Return: {min_var_result['expected_return']:.2%}")
        print(f"Volatility: {min_var_result['volatility']:.2%}")
        print(f"Sharpe Ratio: {min_var_result['sharpe_ratio']:.3f}")
        print("Optimal Weights:")
        for i, symbol in enumerate(symbols):
            print(f"  {symbol}: {min_var_result['weights'][i]:.1%}")
    
    # Performance comparison
    print(f"\n{'OPTIMIZATION RESULTS COMPARISON':-^80}")
    print(f"{'Portfolio':<20} {'Return':<10} {'Volatility':<12} {'Sharpe':<8} {'Improvement'}")
    print("-" * 80)
    
    print(f"{'Equal Weight':<20} {equal_return:<10.2%} {equal_vol:<12.2%} {equal_sharpe:<8.3f} {'Benchmark'}")
    
    if max_sharpe_result['success']:
        sharpe_improvement = ((max_sharpe_result['sharpe_ratio'] / equal_sharpe) - 1) * 100
        print(f"{'Max Sharpe':<20} {max_sharpe_result['expected_return']:<10.2%} {max_sharpe_result['volatility']:<12.2%} {max_sharpe_result['sharpe_ratio']:<8.3f} {sharpe_improvement:+.1f}%")
    
    if min_var_result['success']:
        vol_improvement = ((equal_vol / min_var_result['volatility']) - 1) * 100
        print(f"{'Min Variance':<20} {min_var_result['expected_return']:<10.2%} {min_var_result['volatility']:<12.2%} {min_var_result['sharpe_ratio']:<8.3f} {vol_improvement:+.1f}% vol")
    
    return {
        'equal_weight': {'weights': equal_weights, 'return': equal_return, 'volatility': equal_vol, 'sharpe': equal_sharpe},
        'max_sharpe': max_sharpe_result if max_sharpe_result['success'] else None,
        'min_variance': min_var_result if min_var_result['success'] else None,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'symbols': symbols
    }

def plot_efficient_frontier(optimization_results):
    """Plot the efficient frontier"""
    if optimization_results['max_sharpe'] is None:
        print("Cannot plot efficient frontier - optimization failed")
        return
    
    mean_returns = optimization_results['mean_returns']
    cov_matrix = optimization_results['cov_matrix']
    symbols = optimization_results['symbols']
    
    print("\nGenerating efficient frontier...")
    frontier_data = generate_efficient_frontier(mean_returns, cov_matrix)
    
    plt.figure(figsize=(12, 8))
    
    # Plot efficient frontier
    valid_points = frontier_data[1] > 0  # Remove invalid points
    plt.plot(frontier_data[1][valid_points], frontier_data[0][valid_points], 
             'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot individual assets
    individual_returns = mean_returns * 252
    individual_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
    
    for i, symbol in enumerate(symbols):
        plt.scatter(individual_vols[i], individual_returns[i], 
                   marker='o', s=100, label=symbol)
    
    # Plot optimized portfolios
    equal_weight = optimization_results['equal_weight']
    plt.scatter(equal_weight['volatility'], equal_weight['return'], 
               marker='s', s=200, c='red', label='Equal Weight', edgecolors='black')
    
    if optimization_results['max_sharpe']:
        max_sharpe = optimization_results['max_sharpe']
        plt.scatter(max_sharpe['volatility'], max_sharpe['expected_return'], 
                   marker='^', s=200, c='green', label='Max Sharpe', edgecolors='black')
    
    if optimization_results['min_variance']:
        min_var = optimization_results['min_variance']
        plt.scatter(min_var['volatility'], min_var['expected_return'], 
                   marker='v', s=200, c='orange', label='Min Variance', edgecolors='black')
    
    plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier & Portfolio Optimization', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format axes as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    print(f"\n📊 Displaying efficient frontier plot...")
    plt.show()

def get_user_input():
    """Get user input for analysis"""
    print("=== Stock Correlation, Covariance & Portfolio Optimization ===\n")
    
    # Get stock symbols
    print("Enter stock symbols (separated by commas or spaces):")
    print("Example: AAPL, GOOGL, MSFT, TSLA")
    symbols_input = input("Stock symbols: ").strip()
    
    # Parse symbols
    symbols = []
    for symbol in symbols_input.replace(',', ' ').split():
        symbol = symbol.strip().upper()
        if symbol:
            symbols.append(symbol)
    
    if len(symbols) < 2:
        print("❌ Please enter at least 2 stock symbols.")
        return None, None, None, None
    
    print(f"✅ Analyzing {len(symbols)} stocks: {', '.join(symbols)}")
    
    # Get time period
    print("\nSelect time period:")
    print("1. 1 year")
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
    
    # Get return type
    print("\nSelect return frequency:")
    print("1. Daily returns")
    print("2. Weekly returns")
    print("3. Monthly returns")
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return_type = "daily"
            break
        elif choice == "2":
            return_type = "weekly"
            break
        elif choice == "3":
            return_type = "monthly"
            break
        else:
            print("Please enter 1, 2, or 3.")
    
    # Ask about portfolio optimization
    print("\nInclude portfolio optimization analysis?")
    while True:
        choice = input("(y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            include_optimization = True
            break
        elif choice in ['n', 'no']:
            include_optimization = False
            break
        else:
            print("Please enter 'y' or 'n'.")
    
    return symbols, period, return_type, include_optimization

def main():
    """Main function"""
    try:
        # Get user input
        symbols, period, return_type, include_optimization = get_user_input()
        
        if symbols is None:
            return
        
        print(f"\n{'='*80}")
        print("STARTING CORRELATION & COVARIANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Download data
        price_data, failed_symbols = get_stock_data(symbols, period)
        
        if price_data is None:
            print("❌ No data available for analysis.")
            return
        
        # Update symbols list to exclude failed ones
        successful_symbols = [s for s in symbols if s not in failed_symbols]
        
        if len(successful_symbols) < 2:
            print("❌ Need at least 2 stocks with valid data for correlation analysis.")
            return
        
        # Calculate returns
        returns_data = calculate_returns(price_data, return_type)
        
        # Calculate correlation and covariance matrices
        correlation_matrix = calculate_correlation_matrix(returns_data)
        covariance_matrix = calculate_covariance_matrix(returns_data)
        
        # Print analysis
        print_correlation_analysis(correlation_matrix, successful_symbols)
        print_covariance_analysis(covariance_matrix, successful_symbols)
        
        # Create visualizations
        create_correlation_heatmap(correlation_matrix, 
                                 f"Stock Correlation Matrix ({period.upper()}, {return_type.title()} Returns)")
        create_covariance_heatmap(covariance_matrix,
                                f"Stock Covariance Matrix ({period.upper()}, {return_type.title()} Returns)")
        
        # Portfolio optimization (if requested)
        if include_optimization:
            print(f"\n{'='*80}")
            print("STARTING PORTFOLIO OPTIMIZATION")
            print(f"{'='*80}")
            
            optimization_results = print_portfolio_optimization(returns_data, successful_symbols)
            plot_efficient_frontier(optimization_results)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        if include_optimization:
            print("🎯 Portfolio optimization results show optimal asset allocations.")
            print("💡 Consider the trade-offs between return, risk, and diversification.")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
