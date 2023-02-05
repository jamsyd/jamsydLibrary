import numpy as np
import pandas as pd


def spread_threshold(pnl_event):

    cachePnL = {

        'asofdate':[],
        'pnl':[],
        'strategy':[],
        'positionType':[],

    }

    # name of string to save output
    cache_pnl_str = pnl_event['dataframe'].replace("forecasts","pnl_armaspreadthreshold")
    cache_trades_str = pnl_event['dataframe'].replace("forecasts","tradelevelpnl_armaspreadthreshold")


    # reading in dataframe
    pnl_event['dataframe'] = pd.read_csv(pnl_event['dataframe'],parse_dates=True)

    # calculating the difference
    close_diff    = pnl_event['dataframe']['close'].diff(pnl_event['forecastHorizon'])


    forecast_diff = pnl_event['dataframe']['pointForecast'].diff(pnl_event['forecastHorizon'])

    i = 0
    while i < len(pnl_event['dataframe']) - pnl_event['forecastHorizon']:
        if np.abs(forecast_diff[i+pnl_event['forecastHorizon']]) > pnl_event['threshold'] and ((forecast_diff[i+pnl_event['forecastHorizon']] > 0 and pnl_event['dataframe']['MA_diff_50'][i] > 0) or (forecast_diff[i+pnl_event['forecastHorizon']] <= 0 and pnl_event['dataframe']['MA_diff_50'][i] <= 0)):

            if forecast_diff[i+pnl_event['forecastHorizon']] > 0 and pnl_event['dataframe']['MA_diff_50'][i] > 0:
                for j in range(0,pnl_event['forecastHorizon']):
                    cachePnL['pnl'].append(close_diff[i + j])
                    cachePnL['positionType'].append('long')

            if forecast_diff[i+pnl_event['forecastHorizon']] <= 0 and pnl_event['dataframe']['MA_diff_50'][i] <= 0:
                for j in range(0,pnl_event['forecastHorizon']):
                    cachePnL['pnl'].append(-close_diff[i + j])
                    cachePnL['positionType'].append('short')

        else:

            for j in range(0,pnl_event['forecastHorizon']):
                cachePnL['pnl'].append(0)
                cachePnL['positionType'].append('no_position')
    
        i+=pnl_event['forecastHorizon']

    cachePnL['asofdate'] = pnl_event['dataframe']['asofdate'][pnl_event['forecastHorizon']:].to_list()
    cachePnL['strategy'] = [pnl_event['strategy']]*len(cachePnL['asofdate'])

    # saving dataframes
    pnl_df   = pd.DataFrame(cachePnL)
    trade_df = pnl_df[::pnl_event['forecastHorizon']]

    # cache to csv
    pnl_df.to_csv(cache_pnl_str)
    trade_df.to_csv(cache_trades_str)
