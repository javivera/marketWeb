import os
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Quote

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

async def get_stock_price(symbol, session):
    """Fetch current stock price using TastyTrade's DXLinkStreamer."""
    if not session:
        print("TastyTrade session not available.")
        return None

    try:
        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Quote, [symbol])
            quote = await streamer.get_event(Quote)
            current_price = float((quote.bid_price + quote.ask_price) / 2)
            return current_price
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def get_other_inputs():
    """Get the rest of the user inputs synchronously."""
    premium = float(input('Precio de venta del call: '))
    remaining_days = float(input('Dias restantes: '))
    strike = float(input('Strike: '))
    return premium, remaining_days, strike

async def returnCall():
    # Get ticker first, as it's needed to start the price fetch
    stock_symbol = input('Stock symbol (e.g., AAPL, GGAL): ').strip().upper()
    
    # Get TastyTrade session
    session = get_tastytrade_session()
    
    if not session:
        print("Warning: TastyTrade session not available.")
        # Get other inputs first, then ask for price manually
        premium, remaining_days, strike = get_other_inputs()
        stock_price = float(input('Enter stock price manually: '))
    else:
        # Start fetching the price in a background task
        price_task = asyncio.create_task(get_stock_price(stock_symbol, session))
        
        # Run the blocking input calls in a separate thread to avoid blocking the event loop
        inputs_task = asyncio.to_thread(get_other_inputs)
        
        # Wait for both tasks to complete concurrently
        results = await asyncio.gather(price_task, inputs_task)
        stock_price = results[0]
        premium, remaining_days, strike = results[1]
        
        if stock_price is None:
            stock_price = float(input('Could not fetch price. Enter stock price manually: '))
    
    # Calculate option metrics
    cobertura = (premium / stock_price) * 100
    percentage_move_to_strike = ((strike / stock_price) - 1) * 100
    max_gains = strike - (stock_price - premium)
    ganancia_pct = premium / (stock_price - premium)
    ganancia_anualizada = ganancia_pct * 365 / remaining_days * 100

    print(f'\n{"OPTION ANALYSIS RESULTS":-^60}')
    print(f'Cobertura: {cobertura:.2f}%')
    print(f'Percentage move needed to reach strike: {percentage_move_to_strike:.2f}%')
    print(f'Max Gains: ${max_gains*100:.2f}')
    print(f'Ganancia anualizada: {ganancia_anualizada:.2f}%')
    print(f'{"-"*60}')

asyncio.run(returnCall())
