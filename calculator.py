import yfinance as yf
import pandas as pd

def get_stock_price(symbol):
    """Fetch current stock price from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        current_price = float(data['Close'].iloc[-1])
        print(f"Current price for {symbol}: ${current_price:.2f}")
        return current_price
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def get_historical_returns_for_period(symbol, historical_period, analysis_days):
    """Calculate average returns for specific period based on historical data"""
    try:
        print(f"Fetching {historical_period} historical data for {symbol}...")
        
        data = yf.download(symbol, period=historical_period, interval='1d', progress=False)
        if data.empty or len(data) < analysis_days + 1:
            return None
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Calculate rolling returns for the specified period
        close_prices = data['Close'].copy()
        close_shifted = close_prices.shift(analysis_days)
        period_returns = ((close_prices / close_shifted) - 1) * 100
        
        # Remove NaN values
        period_returns = period_returns.dropna()
        
        if len(period_returns) == 0:
            return None
        
        # Calculate statistics
        avg_return = period_returns.mean()
        median_return = period_returns.median()
        std_return = period_returns.std()
        min_return = period_returns.min()
        max_return = period_returns.max()
        positive_returns = period_returns[period_returns > 0]
        negative_returns = period_returns[period_returns < 0]
        
        # Calculate annualized return
        periods_per_year = 252 / analysis_days  # Number of periods per year
        annualized_return = avg_return * periods_per_year
        
        return {
            'avg_return': avg_return,
            'median_return': median_return,
            'std_return': std_return,
            'min_return': min_return,
            'max_return': max_return,
            'annualized_return': annualized_return,
            'total_observations': len(period_returns),
            'positive_count': len(positive_returns),
            'negative_count': len(negative_returns),
            'prob_positive': len(positive_returns) / len(period_returns) * 100,
            'historical_period': historical_period,
            'analysis_days': analysis_days,
            'periods_per_year': periods_per_year
        }
        
    except Exception as e:
        print(f"Error analyzing {analysis_days}-day returns: {e}")
        return None

def print_period_analysis(symbol, analysis_data):
    """Print period-specific analysis"""
    if analysis_data is None:
        print("Could not analyze period returns.")
        return
    
    print(f"\n{f'{analysis_data["analysis_days"]}-DAY RETURN ANALYSIS FOR {symbol}':-^60}")
    print(f"\n{'RETURN STATISTICS':-^60}")
    print(f"Average {analysis_data['analysis_days']}-day return: {analysis_data['avg_return']:.2f}%")
    print(f"Median {analysis_data['analysis_days']}-day return: {analysis_data['median_return']:.2f}%")
    print(f"Standard deviation: {analysis_data['std_return']:.2f}%")
    print(f"Best {analysis_data['analysis_days']}-day return: {analysis_data['max_return']:.2f}%")
    print(f"Worst {analysis_data['analysis_days']}-day return: {analysis_data['min_return']:.2f}%")
    print(f"\n{'PROBABILITY ANALYSIS':-^60}")
    print(f"Probability of positive return: {analysis_data['prob_positive']:.1f}%")
    print(f"\n{'ANNUALIZED METRICS':-^60}")
    print(f"Annualized return: {analysis_data['annualized_return']:.2f}%")
    print(f"Periods per year: {analysis_data['periods_per_year']:.1f}")
    print(f"{'-'*60}")

def returnPut():
    stock_symbol = input('Stock symbol (e.g., AAPL, GGAL): ').strip().upper()
    stock_price = get_stock_price(stock_symbol)

def get_analysis_period():
    """Get user's choice for analysis period in days"""
    while True:
        try:
            days = int(input("Enter the period in days to analyze (e.g., 30, 60, 90): "))
            if days <= 0:
                print("Please enter a positive number of days.")
                continue
            if days > 365:
                print("Warning: Period is longer than 1 year. This might limit available data.")
            return days
        except ValueError:
            print("Please enter a valid number.")

def returnCall():
    # Collect all user inputs first
    stock_symbol = input('Stock symbol (e.g., AAPL, GGAL): ').strip().upper()
    
    print("\nSelect historical period for analysis:")
    print("1. 1 year")
    print("2. 2 years") 
    print("3. 3 years")
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == "1":
            historical_period = "1y"
            break
        elif choice == "2":
            historical_period = "2y"
            break
        elif choice == "3":
            historical_period = "3y"
            break
        else:
            print("Please enter 1, 2, or 3.")
    
    while True:
        try:
            analysis_days = int(input("Enter the period in days to analyze (e.g., 30, 60, 90): "))
            if analysis_days <= 0:
                print("Please enter a positive number of days.")
                continue
            if analysis_days > 365:
                print("Warning: Period is longer than 1 year. This might limit available data.")
            break
        except ValueError:
            print("Please enter a valid number.")
    
    premium = float(input('Precio de venta del call: '))
    remaining_days = float(input('Dias restantes: '))
    strike = float(input('Strike: '))
    
    # Now process all the data and show results
    print(f"\n{'='*60}")
    print("PROCESSING DATA...")
    print(f"{'='*60}")
    
    # Get stock price
    stock_price = get_stock_price(stock_symbol)
    if stock_price is None:
        stock_price = float(input('Could not fetch price. Enter stock price manually: '))
    
    # Analyze historical returns
    analysis_data = get_historical_returns_for_period(stock_symbol, historical_period, analysis_days)
    print_period_analysis(stock_symbol, analysis_data)
    
    # Calculate option metrics
    cobertura = (premium / stock_price) * 100
    percentage_move_to_strike = ((strike / stock_price) - 1) * 100
    ganancia = strike - (stock_price - premium)
    ganancia_pct = ganancia / (stock_price - premium)
    ganancia_anualizada = ganancia_pct * 365 / remaining_days * 100

    print(f'\n{"OPTION ANALYSIS RESULTS":-^60}')
    print(f'Cobertura: {cobertura:.2f}%')
    print(f'Percentage move needed to reach strike: {percentage_move_to_strike:.2f}%')
    print(f'Ganancia anualizada: {ganancia_anualizada:.2f}%')
    print(f'{"-"*60}')


def returnPut():
    stock_symbol = input('Stock symbol (e.g., AAPL, GGAL): ').strip().upper()
    stock_price = get_stock_price(stock_symbol)
    
    if stock_price is None:
        stock_price = float(input('Could not fetch price. Enter stock price manually: '))
    
    premium = float(input('Precio de venta del put: '))
    remaining_days = float(input('Dias restantes: '))
    strike = float(input('Strike: '))

    cobertura = (((strike - premium) / stock_price ) -1) *100
    
    # Calculate percentage move needed to reach strike (negative for puts)
    percentage_move_to_strike = ((strike / stock_price) - 1) * 100

    ganancia = premium
    ganancia_pct = ganancia / (strike - premium)
    ganancia_anualizada = ganancia_pct * 365 / remaining_days * 100

    print(f'Cobertura: {cobertura:.2f}%')
    print(f'Percentage move needed to reach strike: {percentage_move_to_strike:.2f}%')
    print(f'Ganancia anualizada: {ganancia_anualizada:.2f}%')

returnCall()