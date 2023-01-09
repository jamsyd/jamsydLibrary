import numpy as np
import pandas as pd


def spread_threshold(pnl_event):

    cachePnL = {

        'asofdate':[],
        'pnl':[],
        'strategy':[],

    }

    pnl_event['dataframe'] = pd.read_csv(pnl_event['dataframe'],parse_dates=True)

    close_diff    = pnl_event['dataframe']['close'].diff(pnl_event['forecastHorizon'])
    forecast_diff = pnl_event['dataframe']['pointForecast'].diff(pnl_event['forecastHorizon'])

    i = 0
    while i < len(pnl_event['dataframe']) - pnl_event['forecastHorizon']:

        if np.abs(forecast_diff[i]) > pnl_event['threshold'] and ((forecast_diff[i+pnl_event['forecastHorizon']]>0 and pnl_event['dataframe']['MA_diff_50'][i]>0) or (forecast_diff[i+pnl_event['forecastHorizon']] <= 0 and pnl_event['dataframe']['MA_diff_50'][i]<=0)):

            if forecast_diff[i+pnl_event['forecastHorizon']]>0 and pnl_event['dataframe']['MA_diff_50'][i]>0:
                for j in range(0,pnl_event['forecastHorizon']):
                    cachePnL['pnl'].append(close_diff[i + j - pnl_event['forecastHorizon']])

            if forecast_diff[i+pnl_event['forecastHorizon']] <= 0 and pnl_event['dataframe']['MA_diff_50'][i]<=0:
                for j in range(0,pnl_event['forecastHorizon']):
                    cachePnL['pnl'].append(-close_diff[i + j - pnl_event['forecastHorizon']])

        else:
            for j in range(0,pnl_event['forecastHorizon']):
                cachePnL['pnl'].append(0)

        i+=pnl_event['forecastHorizon']

    cachePnL['asofdate'] = pnl_event['dataframe']['asofdate'][pnl_event['forecastHorizon']:].to_list()
    cachePnL['strategy'] = [pnl_event['strategy']]*len(cachePnL['asofdate'])

    print(len(cachePnL['asofdate']))
    print(len(cachePnL['strategy']))

    pd.DataFrame(cachePnL).to_csv(r'pnl_test.csv')


pnl_event = {

    'forecastHorizon':5,
    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\approved_strategies\corn_arma_ma\output\arma_(3,3)\forecasts_corn_(3, 3)_True_5.csv',
    'threshold':0.01,
    'reinvest':True,
    'strategy':'arma_threshold_ma',
}

spread_threshold(pnl_event)