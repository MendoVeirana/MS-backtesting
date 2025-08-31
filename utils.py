import pandas as pd
import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt

def compute_rsi(series, period=14):
    """
    Computes the RSI (Relative Strength Index) for a given price series.
    Returns a pd.Series of RSI values (same length as the input).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def plot_btc_prices(data, date_col='Date', price_col='Close'):
    """
    Plot BTC prices over time.
    
    Parameters:
    - data: DataFrame containing BTC data.
    - date_col: Column name for dates (default: 'Date').
    - price_col: Column name for prices (default: 'Close').
    """
    # Convert date column to datetime format for better plotting
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(data[date_col], data[price_col], label='BTC Price', linewidth=2)
    plt.title('Bitcoin Price Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.show()


def load_btc_data(start, end):
    """
    Loads historical Bitcoin data from a CSV file.
    We assume the CSV has columns: 'Date' and 'Close'.
    It returns a pandas DataFrame with a DateTime index sorted by Date.
    """
    csv_filename = f'BTC-USD_{start}_{end}.csv'
    df_btc = yf.download('BTC-USD', start=start, end=end, interval="1d")
    df_btc.to_csv(f'BTC-USD_{start}_{end}.csv')        
    return df_btc

# Ladder function
def ladder(delta_pct, INITIAL_MARGIN):
    if delta_pct <= -0.40:
        return 0.03*INITIAL_MARGIN, 0.03*INITIAL_MARGIN
    if delta_pct > 0:
        return 0.01*INITIAL_MARGIN, 0.0
    if delta_pct > -0.05:
        return 0.02*INITIAL_MARGIN, 0.0
    if delta_pct > -0.10:
        return 0.03*INITIAL_MARGIN, 0.0
    if delta_pct > -0.15:
        return 0.04*INITIAL_MARGIN, 0.0
    if delta_pct > -0.20:
        return 0.05*INITIAL_MARGIN, 0.0
    return 0.0,0.0
