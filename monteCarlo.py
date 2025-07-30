import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def run_monte_carlo_simulation(stock_symbol, min_price, days=20, num_simulations=5000, historical_period="1y"):
    """
    Run Monte Carlo simulation to predict if stock price will be above minimum threshold.

    Parameters:
        stock_symbol (str): Stock symbol for which the simulation is performed.
        min_price (float): Minimum price threshold to analyze.
        days (int): Number of days to predict.
        num_simulations (int): Number of Monte Carlo simulations to run.
        historical_period (str): Historical data period to use ('1y', '3y', or '5y').
    """
    # Fetch historical stock prices (fixed to 1d interval)
    print(f"Fetching {historical_period} of historical data for {stock_symbol}...")
    try:
        historical_data = yf.download(stock_symbol, period=historical_period, interval="1d", progress=False)
        
        # Force download to ensure we get all columns
        if historical_data.empty or 'Close' not in historical_data.columns:
            historical_data = yf.Ticker(stock_symbol).history(period=historical_period, interval="1d")
            
    except Exception as e:
        raise ValueError(f"Failed to download data for {stock_symbol}: {e}")
    
    if historical_data.empty:
        raise ValueError(f"No data found for symbol {stock_symbol}. Please check the symbol and try again.")
    
    # Ensure we have the Close column
    if 'Close' not in historical_data.columns:
        raise ValueError(f"No 'Close' price data found for {stock_symbol}. Available columns: {list(historical_data.columns)}")
    
    
    # Handle multi-level columns if they exist
    if isinstance(historical_data.columns, pd.MultiIndex):
        historical_data.columns = historical_data.columns.droplevel(1)
    
    latest_price = float(historical_data['Close'].iloc[-1])
    
    # Calculate required return to reach minimum price
    required_return = (min_price / latest_price - 1) * 100
    
    # Extract closing prices and calculate historical daily returns
    closing_prices = historical_data['Close'].dropna().values  # Remove any NaN values
    dates = historical_data.index
  

    # Calculate daily returns more robustly
    returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Check if returns calculation worked
    
    # Remove any NaN or infinite values more carefully
    finite_mask = np.isfinite(returns)
    returns = returns[finite_mask]
    
    # Estimate probability density function (PDF) using kernel density estimation (KDE) on historical returns
    try:
        kde = gaussian_kde(returns)
    except Exception as e:
        raise ValueError(f"Failed to create KDE: {e}. Returns shape: {returns.shape}")

    # Perform Monte Carlo simulation multiple times
    print(f"\nRunning {num_simulations:,} Monte Carlo simulations...")
    simulated_prices_all = []
    above_threshold_count = 0
    
    # Simple progress indicator
    progress_points = [int(num_simulations * i / 10) for i in range(1, 11)]
    
    for i in range(num_simulations):
        if i in progress_points:
            progress = int((i / num_simulations) * 100)
            print(f"Progress: {progress}%")
        
        final_price = monte_carlo_simulation(closing_prices, kde, days)
        simulated_prices_all.append(final_price)
        
        # Count how many simulations end above the minimum threshold
        if final_price >= min_price:
            above_threshold_count += 1
    

    # Calculate probability of being above threshold
    probability_above_threshold = above_threshold_count / num_simulations
    
    # Calculate statistics
    mean_price = float(np.mean(simulated_prices_all))
    max_price = float(max(simulated_prices_all))
    min_simulated_price = float(min(simulated_prices_all))
    std_price = float(np.std(simulated_prices_all))
    median_price = float(np.median(simulated_prices_all))
    
    # Calculate percentiles
    percentile_5 = float(np.percentile(simulated_prices_all, 5))
    percentile_25 = float(np.percentile(simulated_prices_all, 25))
    percentile_75 = float(np.percentile(simulated_prices_all, 75))
    percentile_95 = float(np.percentile(simulated_prices_all, 95))
    
    # Separate above and below threshold prices for analysis
    above_threshold_prices = [p for p in simulated_prices_all if p >= min_price]
    below_threshold_prices = [p for p in simulated_prices_all if p < min_price]
    
    # Print results
    print(f"\n{'='*70}")
    print(f"MONTE CARLO THRESHOLD ANALYSIS FOR {stock_symbol}")
    print(f"Using {historical_period} of historical data ({len(closing_prices)} trading days)")
    print(f"{'='*70}")
    
    print(f"\n{'MAIN RESULT':-^70}")
    print(f"🎯 PROBABILITY OF BEING ABOVE ${min_price:.2f}: {probability_above_threshold:.2%}")
    
    # Probability assessment
    if probability_above_threshold >= 0.8:
        likelihood = "VERY HIGH"
        emoji = "🚀"
    elif probability_above_threshold >= 0.6:
        likelihood = "HIGH"
        emoji = "📈"
    elif probability_above_threshold >= 0.4:
        likelihood = "MODERATE"
        emoji = "⚖️"
    elif probability_above_threshold >= 0.2:
        likelihood = "LOW"
        emoji = "📉"
    else:
        likelihood = "VERY LOW"
        emoji = "🔻"
    
    print(f"Assessment: {emoji} {likelihood} likelihood of reaching threshold")
    
    print(f"\n{'PRICE STATISTICS':-^70}")
    print(f"  • Mean predicted price: ${mean_price:.2f}")
    print(f"  • Median predicted price: ${median_price:.2f}")
    print(f"  • Standard deviation: ${std_price:.2f}")
    print(f"  • Minimum predicted price: ${min_simulated_price:.2f}")
    print(f"  • Maximum predicted price: ${max_price:.2f}")
    
    print(f"\n{'RISK ANALYSIS':-^70}")
    downside_risk = (percentile_5 - latest_price) / latest_price * 100
    upside_potential = (percentile_95 - latest_price) / latest_price * 100
    
    print(f"  • 5th percentile (worst 5%): ${percentile_5:.2f} ({downside_risk:.1f}%)")
    print(f"  • 25th percentile: ${percentile_25:.2f}")
    print(f"  • 75th percentile: ${percentile_75:.2f}")
    print(f"  • 95th percentile (best 5%): ${percentile_95:.2f} ({upside_potential:.1f}%)")
    
    print(f"\n{'THRESHOLD ANALYSIS':-^70}")
    print(f"  • Simulations above threshold: {above_threshold_count:,} / {num_simulations:,}")
    print(f"  • Simulations below threshold: {num_simulations - above_threshold_count:,} / {num_simulations:,}")
    
    if above_threshold_prices:
        avg_above = float(np.mean(above_threshold_prices))
        print(f"  • Average price when above threshold: ${avg_above:.2f}")
    
    if below_threshold_prices:
        avg_below = float(np.mean(below_threshold_prices))
        print(f"  • Average price when below threshold: ${avg_below:.2f}")
    
    # Trading insights
    print(f"\n{'TRADING INSIGHTS':-^70}")
    if latest_price >= min_price:
        distance_pct = (latest_price - min_price) / latest_price * 100
        print(f"✅ Currently above threshold by {distance_pct:.1f}%")
        if probability_above_threshold >= 0.7:
            print("💡 High probability of staying above threshold")
        else:
            print("⚠️  Risk of falling below threshold")
    else:
        needed_gain = ((min_price / latest_price) - 1) * 100
        print(f"📈 Need {needed_gain:.1f}% gain to reach threshold")
        if probability_above_threshold >= 0.5:
            print("💡 Good probability of reaching threshold")
        else:
            print("⚠️  Low probability of reaching threshold")
    
    # Confidence levels
    print(f"\n{'CONFIDENCE LEVELS':-^70}")
    if percentile_10 := float(np.percentile(simulated_prices_all, 10)):
        if percentile_10 >= min_price:
            print("✅ 90% confidence: Above threshold")
        else:
            print("❌ 90% confidence: Below threshold")
    
    if percentile_25 >= min_price:
        print("✅ 75% confidence: Above threshold")
    else:
        print("❌ 75% confidence: Below threshold")
    
    if median_price >= min_price:
        print("✅ 50% confidence: Above threshold")
    else:
        print("❌ 50% confidence: Below threshold")
    
    print(f"\n{'='*70}")
    
    # Create and show a simple graph
    create_simulation_graph(historical_data, closing_prices, kde, days, min_price, stock_symbol, num_sample_paths=5)
    
    return {
        'probability_above_threshold': probability_above_threshold,
        'mean_price': mean_price,
        'median_price': median_price,
        'std_price': std_price,
        'current_price': latest_price,
        'min_price': min_price,
        'simulations_above': above_threshold_count,
        'total_simulations': num_simulations
    }

def monte_carlo_simulation(prices, kde, days):
    """
    Perform Monte Carlo simulation to predict future stock prices.

    Parameters:
        prices (numpy.ndarray): Array containing historical stock prices.
        kde (scipy.stats.gaussian_kde): Kernel density estimator for generating samples.
        days (int): The number of days to predict.

    Returns:
        float: The final simulated price after the specified number of days.
    """
    current_price = float(prices[-1])  # Start with the last known price

    for _ in range(days):
        try:
            # Generate a sample from the KDE
            sampled_return = kde.resample(1)
            
            # Handle different array shapes that might be returned
            if isinstance(sampled_return, np.ndarray):
                if sampled_return.ndim == 2:
                    sampled_return = float(sampled_return[0, 0])
                else:
                    sampled_return = float(sampled_return[0])
            else:
                sampled_return = float(sampled_return)
            
            # Apply the return to get new price
            current_price = current_price * (1 + sampled_return)
            
            # Sanity check: prevent extreme values
            if current_price <= 0 or current_price > 1e6:
                # Reset to a reasonable value if we get extreme results
                current_price = float(prices[-1])
                
        except Exception:
            # Continue with current price if there's an error
            continue
    
    return current_price

def create_simulation_graph(historical_data, closing_prices, kde, days, min_price, stock_symbol, num_sample_paths=5):
    """
    Create a graph showing historical prices and sample Monte Carlo simulation paths.
    """
    try:
        import matplotlib.dates as mdates
        from datetime import datetime, timedelta
        
        plt.figure(figsize=(12, 8))
        
        # Show only the last 30 days of historical data for better context
        last_30_days = historical_data.tail(30)
        
        # Plot the last 30 days of historical prices
        plt.plot(last_30_days.index, last_30_days['Close'], 
                label='Historical Prices (Last 30 Days)', color='blue', linewidth=2)
        
        # Get the last date and price for starting simulations
        last_date = historical_data.index[-1]
        last_price = float(closing_prices[-1])
        
        # Create future dates for simulation paths
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        
        # Generate sample simulation paths
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(min(num_sample_paths, len(colors))):
            # Run one simulation path
            simulation_prices = [last_price]
            current_price = last_price
            
            for _ in range(days):
                # Generate a sample from the KDE
                sampled_return = kde.resample(1)
                if isinstance(sampled_return, np.ndarray):
                    if sampled_return.ndim == 2:
                        sampled_return = float(sampled_return[0, 0])
                    else:
                        sampled_return = float(sampled_return[0])
                else:
                    sampled_return = float(sampled_return)
                
                current_price = current_price * (1 + sampled_return)
                simulation_prices.append(current_price)
            
            # Plot this simulation path
            all_dates = [last_date] + future_dates
            plt.plot(all_dates, simulation_prices, 
                    color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5,
                    label=f'Simulation Path {i+1}')
        
        # Add threshold line
        plt.axhline(y=min_price, color='black', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Threshold: ${min_price:.2f}')
        
        # Add a vertical line to separate historical and simulated data
        plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.8, linewidth=1,
                   label='Today')
        
        # Formatting
        plt.title(f'{stock_symbol} - Recent History (30 days) vs Monte Carlo Simulation Paths ({days} days)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Adjust layout and show
        plt.tight_layout()
        print(f"\n📊 Displaying graph with last 30 days of history + {num_sample_paths} sample simulation paths...")
        plt.show()
        
    except Exception as e:
        print(f"Could not create graph: {e}")
        print("Graph display skipped.")

def get_user_input():
    """Get user input for simulation parameters."""
    print("=== Monte Carlo Stock Price Threshold Analysis ===\n")
    
    # Get stock symbol
    stock_symbol = input("Enter stock symbol (e.g., AAPL, TSLA, GOOGL): ").strip().upper()
    
    # Get minimum price threshold
    while True:
        try:
            min_price = float(input("Enter minimum price threshold: $"))
            if min_price <= 0:
                print("Please enter a positive price.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get historical data period
    print("\nSelect historical data period:")
    print("1. 1 year (recommended for most stocks)")
    print("2. 3 years (better for long-term analysis)")
    print("3. 5 years (maximum historical context)")
    
    while True:
        period_choice = input("Enter your choice (1, 2, or 3): ").strip()
        if period_choice == "1":
            historical_period = "1y"
            break
        elif period_choice == "2":
            historical_period = "3y"
            break
        elif period_choice == "3":
            historical_period = "5y"
            break
        else:
            print("Please enter 1, 2, or 3.")
    
    # Get number of days to predict
    while True:
        try:
            days = int(input("Enter number of days to predict (default: 20): ") or "20")
            if days <= 0:
                print("Please enter a positive number of days.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get number of simulations
    while True:
        try:
            num_simulations = int(input("Enter number of simulations (default: 5000): ") or "5000")
            if num_simulations <= 0:
                print("Please enter a positive number of simulations.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    return stock_symbol, min_price, days, num_simulations, historical_period

if __name__ == "__main__":
    try:
        # Single stock threshold analysis
        stock_symbol, min_price, days, num_simulations, historical_period = get_user_input()
        print(f"\nRunning threshold analysis for {stock_symbol} using {historical_period} of data...")
        result = run_monte_carlo_simulation(stock_symbol, min_price, days, num_simulations, historical_period)
        print(f"\nAnalysis complete!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your inputs and try again.")

