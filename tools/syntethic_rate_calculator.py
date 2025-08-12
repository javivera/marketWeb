"""
Option Yield Calculator - Calculates annualized cost of options
"""

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Quote

# Load environment variables from .env file
load_dotenv()

def get_tastytrade_session():
    """Creates and returns a TastyTrade session."""
    tasty_user = os.environ.get('TASTY_USER')
    tasty_pass = os.environ.get('TASTY_PASS')

    if not tasty_user or not tasty_pass:
        print("Warning: TASTY_USER and TASTY_PASS environment variables not set.")
        return None

    try:
        session = Session(tasty_user, tasty_pass)
        return session
    except Exception as e:
        print(f"Error creating TastyTrade session: {e}")
        return None

async def get_stock_price(ticker, session):
    """Fetch current stock price using TastyTrade's DXLinkStreamer."""
    if not session:
        print("TastyTrade session not available.")
        return None

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, [ticker])
        quote = await streamer.get_event(Quote)
        return float((quote.bid_price + quote.ask_price) / 2)
            
def calculate_annualized_yield(stock_price, premium_paid, strike_price, days):
    """
    Calculate annualized yield for an option strategy.
    
    Args:
        stock_price: Current stock price
        premium_paid: Premium you paid for the option
        strike_price: Strike price of the option
        days: Days until expiration
        
    Returns:
        Annualized yield as a percentage
    """
    # Calculate the adjusted premium (premium minus intrinsic value)
    intrinsic_value = max(0, stock_price - strike_price)  # For call options
    adjusted_premium = premium_paid - intrinsic_value
    
    # Calculate the raw return rate
    raw_return = adjusted_premium / stock_price
    
    # Annualize the return
    annualized_return = (1 + raw_return) ** (365 / days) - 1
    return annualized_return * 100

def get_other_inputs():
    """Get the rest of the user inputs synchronously."""
    strike_price = float(input("Strike Price: ").strip())
    premium_paid = float(input("Premium Paid: ").strip())
    days = int(input("Days to Expiration: ").strip())
    
    if days <= 0:
        raise ValueError("Days to expiration must be greater than 0")
    
    return strike_price, premium_paid, days

async def main():
    """Calculate option yield with ticker lookup."""
    session = get_tastytrade_session()
    if not session:
        print("Exiting: Cannot proceed without a TastyTrade session.")
        return

    try:
        # Get ticker first, as it's needed to start the price fetch.
        ticker = input("Ticker: ").strip().upper()

        # Start fetching the price in a background task.
        price_task = asyncio.create_task(get_stock_price(ticker, session))

        # Run the blocking input calls in a separate thread to avoid blocking the event loop.
        inputs_task = asyncio.to_thread(get_other_inputs)

        # Wait for both tasks to complete concurrently.
        results = await asyncio.gather(price_task, inputs_task)
        stock_price = results[0]
        strike_price, premium_paid, days = results[1]
        
        if stock_price is None:
            print(f"Could not fetch price for {ticker}. Exiting.")
            return
        
        print(f"\nCurrent price for {ticker}: ${stock_price:.2f}")
        print(f"Strike Price: ${strike_price:.2f}")
        print(f"Premium Paid: ${premium_paid:.2f}")
        print(f"Days to Expiration: {days}")
        
        # Calculate intrinsic value and adjusted premium
        intrinsic_value = max(0, stock_price - strike_price)
        adjusted_premium = premium_paid - intrinsic_value
        
        print(f"Intrinsic Value: ${intrinsic_value:.2f}")
        print(f"Adjusted Premium (after subtracting intrinsic value): ${adjusted_premium:.2f}")
        
        # Calculate and print result
        annualized_yield = calculate_annualized_yield(stock_price, premium_paid, strike_price, days)
        print(f"\nAnnualized Return: {annualized_yield:.2f}%")
        
    except ValueError:
        print("Error: Invalid input.")
    except KeyboardInterrupt:
        print("\nCancelled.")

if __name__ == "__main__":
    asyncio.run(main())
