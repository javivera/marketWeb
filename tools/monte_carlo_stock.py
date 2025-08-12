import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def monte_carlo_simulation(initial_price, daily_return, daily_volatility, num_trading_days, num_simulations):
    """
    Performs a Monte Carlo simulation for a single stock.

    Args:
        initial_price (float): The starting price of the stock.
        daily_return (float): The average daily return of the stock (e.g., 0.0005 for 0.05%).
        daily_volatility (float): The daily volatility (standard deviation) of the stock.
        num_trading_days (int): The number of trading days to simulate.
        num_simulations (int): The number of simulation paths to generate.

    Returns:
        pandas.DataFrame: A DataFrame where each column is a simulation path
                          and rows represent daily prices.
    """
    price_paths = np.zeros((num_trading_days, num_simulations))
    price_paths[0] = initial_price

    for s in range(num_simulations):
        for t in range(1, num_trading_days):
            # Brownian motion component
            st_dev = daily_volatility * np.random.normal(0, 1)
            # Geometric Brownian Motion formula
            price_paths[t, s] = price_paths[t-1, s] * np.exp(daily_return + st_dev)

    return pd.DataFrame(price_paths)

def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Fetches adjusted close price history for a given stock symbol using yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch 'Adj Close' prices
        df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                            interval='1d',
                            auto_adjust=True) # Explicitly set auto_adjust to True
        if df.empty:
            return pd.Series()
        
        ser = df['Close']
        ser.index = pd.to_datetime(ser.index).date
        ser.name = symbol.upper()
        return ser
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.Series()

if __name__ == "__main__":
    stock_ticker = input("Enter the stock ticker symbol (e.g., AAPL, MSFT): ").upper()
    
    while True:
        try:
            num_trading_days_input = input("Enter the number of trading days to simulate (e.g., 252 for 1 year): ")
            num_trading_days = int(num_trading_days_input)
            if num_trading_days <= 0:
                print("Number of trading days must be a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer for the number of trading days.")

    num_simulations = 10000  # Number of simulation paths

    # Define date range for historical data (last 1 year)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    try:
        # Fetch historical data using the new function
        stock_prices = fetch_stock_data(stock_ticker, start_date, end_date)

        if stock_prices.empty:
            print(f"Could not retrieve sufficient data for {stock_ticker}. Please check the ticker symbol or try again later.")
        else:
            initial_price = stock_prices.iloc[-1]
            
            # Calculate daily returns and volatility
            daily_returns = stock_prices.pct_change().dropna()
            if daily_returns.empty:
                print(f"Not enough data to calculate daily returns for {stock_ticker}.")
                exit()

            daily_return = daily_returns.mean()
            daily_volatility = daily_returns.std()

            print(f"Running Monte Carlo simulation for {stock_ticker} with:")
            print(f"  Initial Price (last available): ${initial_price:.2f}")
            print(f"  Daily Return: {daily_return:.4f} (approx. {daily_return*100:.2f}%)")
            print(f"  Daily Volatility: {daily_volatility:.4f} (approx. {daily_volatility*100:.2f}%)")
            print(f"  Number of Trading Days: {num_trading_days}")
            print(f"  Number of Simulations: {num_simulations}\n")

            simulated_prices = monte_carlo_simulation(
                initial_price, daily_return, daily_volatility, num_trading_days, num_simulations
            )

            # Calculate and print some statistics
            final_prices = simulated_prices.iloc[-1]
            print(f"\nMean final price: ${final_prices.mean():.2f}")
            print(f"Standard deviation of final prices: ${final_prices.std():.2f}")
            print(f"95% Confidence Interval for final price: "
                  f"(${np.percentile(final_prices, 2.5):.2f}, ${np.percentile(final_prices, 97.5):.2f})")
            
            # Calculate mean of the lower 50% of final prices (mean of "bad prices")
            sorted_final_prices = np.sort(final_prices)
            lower_50_percent_prices = sorted_final_prices[:len(sorted_final_prices) // 2]
            mean_bad_prices = np.mean(lower_50_percent_prices)
            print(f"Mean of lower 50% of final prices (Mean of 'Bad Prices'): ${mean_bad_prices:.2f}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure you have an active internet connection and the stock ticker is valid.")
