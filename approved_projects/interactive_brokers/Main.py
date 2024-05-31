"""
    Date: 05/28/2024
    Author: James Stanley
    Description: Momentum algo
"""

import os
import glob

import numpy as np
import pandas as pd

import yfinance as yf

import multiprocessing
from multiprocessing import get_context



SPY_PATH    = r'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\tickers\SPY_INDEX_COMPANIES.csv'
TICKER_PATH = r'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\spy_tickers'

def download_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        hist['Ticker'] = ticker  # Add ticker column for easier identification later
        return hist
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return pd.DataFrame()

##################################################
# Download data
##################################################

ticker_data = pd.read_csv(SPY_PATH)
sp500_tickers = ticker_data['tickers'].to_list()

pool = get_context("spawn").Pool(processes = multiprocessing.cpu_count())

# Download historical data for all tickers and save to separate CSV files
for ticker in sp500_tickers:
    print(f"Downloading data for {ticker}...")
    data = download_data(ticker)
    if not data.empty:
        file_name = TICKER_PATH + rf"\{ticker}_historical_data.csv"
        data.to_csv(file_name)
        print(f"Data for {ticker} saved to {file_name}")

################################################################
# Begin implementing algo
################################################################



# use glob to get all the csv files 
# path      = r'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\spy_tickers'
# csv_files = glob.glob(os.path.join(path, "*.csv"))

# settings = [18,50,200]

# position_dict = {'position_id':[],
#                  'return':[],
#                  'std':[],
#                  'days_in_trade':[],
#                  'ema1':[],
#                  'ema2':[],
#                  'ema3':[],
#                  }

# m = 0
# historical_data = []
# for file in csv_files:

#     time_series = pd.read_csv(file,parse_dates=True,index_col='Date')

#     # Calculate the EWMA with a span of 20
#     time_series['ema']  = time_series['Close'].ewm(span=settings[0], adjust=False).mean()
#     time_series['ema1'] = time_series['Close'].ewm(span=settings[1], adjust=False).mean()
#     time_series['ema2'] = time_series['Close'].ewm(span=settings[2], adjust=False).mean()

#     # Create our PnL vector
#     time_series['pnl'] = time_series['Close'].diff(1)

#     ath_vector = []

#     i = 0
#     while i < len(time_series):
#         ath_vector.append(time_series['Close'].iloc[0:i].max())
#         i+=1

#     time_series['ath'] = ath_vector
#     time_series['ath_drawdown'] = time_series['Close']/time_series['ath']

#     time_series['position'] = np.where((time_series['ema'] > time_series['ema1']) & 
#                                     (time_series['ema1'] > time_series['ema2']) & (time_series['ath_drawdown'] > 0.95), 1,0) 
#                                     # np.where((time_series['ema'] < time_series['ema1']) & 
#                                     #         (time_series['ema1'] < time_series['ema2']) & (time_series['ath_drawdown'] < 0.7) & (time_series['ath_drawdown'] > 0.3), -1, 0))

#     # Create a position identifier
#     i               = 0
#     vector          = []
#     position_size   = []
#     entry_price     = []
#     initial_capital = 100000
#     while i < len(time_series):

#         if time_series['position'][i] == 0:
#             vector.append(0)
#             position_size.append(0)
#             entry_price.append(0)

#             i += 1  # Move to the next index
#         else:
#             # Start j = 0 so that we only account for pnl once position is taken
#             j = 1
#             vector.append(0)
#             position_size.append(0)
#             entry_price.append(time_series['Close'][i])
#             position_identifier = f"{file}_{i}"
#             while i + j < len(time_series) and time_series['position'][i + j] == 1:
#                 if j == 1:
#                     size = np.floor(initial_capital/time_series['Close'][i])
#                 entry_price.append(time_series['Close'][i])
#                 vector.append(position_identifier)
#                 position_size.append(size)
#                 j += 1

#             i = j + i

#     time_series['vector']        = vector
#     time_series['position_size'] = position_size
#     time_series['entry_price']   = entry_price

#     # Create PnL Vector
#     time_series['pnl_vector'] = time_series['pnl']*time_series['position']*time_series['position_size']

#     trade_summary = time_series.copy()


#     for position_id in trade_summary['vector'].unique()[1:]:
#         position_dict['position_id'].append(position_id)
#         position_dict['return'].append(trade_summary[trade_summary['vector'] == position_id]['pnl'].sum()/trade_summary[trade_summary['vector'] == position_id]['entry_price'][0]*100)
#         position_dict['std'].append(np.std(trade_summary[trade_summary['vector'] == position_id]['pnl']/trade_summary[trade_summary['vector'] == position_id]['entry_price'][0]))
#         position_dict['days_in_trade'].append(len(trade_summary[trade_summary['vector'] == position_id]))
#         position_dict['ema1'].append(settings[0])
#         position_dict['ema2'].append(settings[1])
#         position_dict['ema3'].append(settings[2])

#     historical_data.append(trade_summary)

#     m+=1
#     print(m)

#     if m > 25:
#         break

        