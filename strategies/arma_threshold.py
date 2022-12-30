import numpy as np
import pandas as pd


def spread_threshold(pnl_event):

    cachePnL = {

        'asofdate':[],
        'pnl':[],

    }

    pnl_event['dataframe'] = pd.read_csv(pnl_event['dataframe'],parse_dates=True)

    close_diff    = pnl_event['dataframe']['close'].diff(pnl_event['forecastHorizon'])
    forecast_diff = pnl_event['dataframe']['pointForecast'].diff(pnl_event['forecastHorizon'])

    i = pnl_event['forecastHorizon']
    while i < len(pnl_event['dataframe']):

        if np.abs(forecast_diff[i]) > pnl_event['threshold']:

            if forecast_diff[i] > 0:
                for j in range(0,pnl_event['forecastHorizon']):
                    cachePnL['pnl'].append(close_diff[i + j - pnl_event['forecastHorizon']])

            if forecast_diff[i] < 0:
                for j in range(0,pnl_event['forecastHorizon']):
                    cachePnL['pnl'].append(-close_diff[i + j - pnl_event['forecastHorizon']])

        else:
            for j in range(0,pnl_event['forecastHorizon']):
                cachePnL['pnl'].append(0)

        i+=pnl_event['forecastHorizon']

    cachePnL['asofdate'] = pnl_event['dataframe']['asofdate'][pnl_event['forecastHorizon']:].to_list()

    pd.DataFrame(cachePnL).to_csv('pnl.csv')


pnl_event = {

    'forecastHorizon':5,
    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\forecasts.csv',
    'threshold':1,
    'reinvest':True,
}

spread_threshold(pnl_event)