import pytz
import numpy as np 
import pandas as pd 
from datetime import datetime

def identify_triangles(timeseries):

    timeseries['triangle'] = None
    timeseries['pnl']      = timeseries['Close'].diff(1)

    i = 0
    while i < len(timeseries) - 4:

        # Contracting Triangle
        if (timeseries['High'].iloc[i - 1] - timeseries['High'].iloc[i-2] < 0) and (timeseries['High'].iloc[i-2] - timeseries['High'].iloc[i-3] < 0) and (timeseries['Low'].iloc[i - 1] - timeseries['Low'].iloc[i-2] < 0) and (df['Low'].iloc[i-2] - timeseries['Low'].iloc[i-3] < 0):

            timeseries['triangle'].iloc[i-3] = "Triangle"
            timeseries['triangle'].iloc[i-2] = "Triangle"
            timeseries['triangle'].iloc[i-1] = "Triangle"

            timeseries['triangle'].iloc[i] = "BUY"
            timeseries['triangle'].iloc[i+1] = "HOLD"
            timeseries['triangle'].iloc[i+2] = "HOLD"
            timeseries['triangle'].iloc[i+3] = "HOLD"

        i+=1

    return timeseries

def select_data(ticker):

    # Example DataFrame with timezone-aware index
    df = pd.read_csv(rf'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\spy_tickers\{ticker}_historical_data.csv', index_col='Date', parse_dates=True)

    # Offset-aware datetimes for comparison
    start_date = datetime(2008, 1, 1, tzinfo=pytz.UTC)
    end_date   = datetime(2020, 1, 1, tzinfo=pytz.UTC)

    # Filter the DataFrame
    filtered_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    return filtered_df


for ticker in ['AAPL','AMZN']:

    input_data = select_data(ticker=ticker)
    identify_triangles(timseries=input_data)

    