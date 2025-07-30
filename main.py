import yfinance as yf
import pandas as pd

def process_galicia_stock(ticker,period='1y', num_days=3):
    """
    Processes GGAL stock data and calculates the mean of the 'Special' column.
    
    Parameters:
    - period (str): The time period for which to download stock data (e.g., '1y', '2y').
    - num_days (int): The number of days to check for consecutive 0s in the 'Trend' column.
    
    Returns:
    - df_filtered (DataFrame): The filtered DataFrame after processing.
    - mean_special (float): The mean of the 'Special' column after replacing -1 with 0.
    - df2 (DataFrame): The DataFrame with all data including calculated columns.
    - len(df): Total number of rows in the DataFrame.
    """
    # Fetch GGAL stock data
    stock_data = yf.download(ticker, period=period, interval='1d', progress=False)

    # Create a DataFrame with just the 'Close' column
    df = pd.DataFrame(stock_data['Close'])

    # Create a 'Trend' column that is 1 if Close is higher than the previous day, 0 otherwise
    df['Trend'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['Absolute Difference'] = abs(df['Close'] - df['Close'].shift(1))
    df['Absolute Percentage Change'] = (df['Absolute Difference'] / df['Close'].shift(1)).abs() * 100

    # Create the 'Special' column
    conditions = [(df['Trend'].shift(i) == 0) for i in range(1, num_days)]

    # Check if all conditions are true across the specified days
    df['Special'] = df['Trend'] == 1
    df['Special'] &= pd.concat(conditions, axis=1).all(axis=1)
    df['Special'] = df['Special'].astype(int)

    # For the case where the last `num_days` and the current day all have 0 in 'Trend', set 'Special' to -1
    df.loc[(df['Trend'] == 0) & pd.concat(conditions, axis=1).all(axis=1), 'Special'] = -1
    df['Special * Absolute Difference'] = df['Absolute Difference'] * df['Special']
    df['Special * Absolute PCT Change'] = df['Absolute Percentage Change'] * df['Special']

    df2 = df[df['Special'] != 0].copy()
    df_filtered = df[df['Special'] != 0].copy()  # Use .copy() to avoid chained assignment

    # Replace -1 with 0 in the 'Special' column without inplace=True
    df_filtered['Special'] = df_filtered['Special'].replace(-1, 0)

    # Calculate the mean of the 'Special' column after replacing -1 with 0
    mean_special = df_filtered['Special'].mean()

    return df_filtered, mean_special, df2, len(df)

# Example of using the function
def loop(ticker):
    for l in ['1y', '2y']:
        print(f"--------------------------- {ticker} Periodo: {l} ---------------------------")
        for j in range(2, 6):
            df_filtered, mean_special, df, len_df = process_galicia_stock(ticker,period=l, num_days=j)  # Example for checking the last 5 days
            print(f"Cantidad de dias para atrasar: {j-1}")
            print(f"Cantidad de apariciones del patron en los ultimos {len(df_filtered)} de total {len_df}")
            print(f"Probabilidad de patron exitoso: {mean_special}")
            
            # Calculate the compounded return from the 'Special * Absolute PCT Change'
            compounded_return = (1 + df['Special * Absolute PCT Change'] / 100).prod() - 1
            
            # Print the compounded return as a percentage
            print(f"Compounded Return: {compounded_return * 100:.2f}%")
            print(df["Special * Absolute PCT Change"].sum())

def main():
    for j in ['GGAL','YPF','TGS','AAPL','SPY']:
        print(j)
        loop(j)

main()
