

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def hitrate(pnl):

    up = 0
    down = 0
    for p in pnl:

        if p < 0:
            down+=1
        if p >= 0:
            up+=1

    return up / (up + down)


def relativepnl(pnl):

    up = 0
    down = 0
    for p in pnl:

        if p < 0:
            down+=np.abs(p)
        if p >= 0:
            up+=np.abs(p)
    return up / (down)

def realized_volatility(returns):
    return (np.log(returns['pnl'].cumsum()/returns['pnl'].cumsum().shift(1))).dropna(axis=0).rolling(window=30).std(ddof=0)*np.sqrt(252)

def get_positions(forecast_df,pnl_df):

    merge_df = pnl_df.merge(forecast_df, left_on='asofdate', right_on='asofdate')

    return merge_df[(merge_df['forecastday'] == 5)][['pnl','pointForecast','forecastday','product_name']]


def pnl_plots(forecast_df,position_df,pnl_df):

    # Plot PnL
    plt.figure(figsize=(45,15))

    plt.plot(pnl_df['pnl'])
    plt.savefig(r'pnl.jpg')

    plt.show()

    # Cumulative PnL
    cumulative_pnl=pnl_df['pnl'].cumsum()/forecast_df['close'][0]

    plt.figure(figsize=(45,15))

    plt.title('Profit and Loss',fontsize=45)
    plt.plot(cumulative_pnl)

    plt.savefig(r'cumulative_pnl.jpg')

    plt.show()

    # Positions histogram
    plt.figure(figsize=(45,15))

    plt.hist(position_df['pnl'][position_df['pnl'] != 0],bins=500)

    plt.savefig(r'pnl_histogram.jpg')
    plt.show()

    # Plotting Forecasts and moving average against price
    plt.figure(figsize=(45,15))

    plt.plot(forecast_df['close'])
    plt.plot(forecast_df['pointForecast'])
    plt.plot(forecast_df['MA_50'])

    plt.legend(['close','pointForecast'],fontsize=45)

    plt.savefig(r'plot_forecasts.jpg')

    plt.show()

    return None


def performance_frame(position_df):

    from scipy.stats import skew, kurtosis

    performance_dict = {

        'No._Trades':[len(position_df['pnl'])],
        'relative_Pnl':[relativepnl(position_df['pnl'])],
        'Hit_Rate':[hitrate(position_df['pnl'])],
        'Minimum':[np.min(position_df['pnl'])],
        'Maximum':[np.max(position_df['pnl'])],
        'Mean':[np.mean(position_df['pnl'])],
        'Variance':[np.var(position_df['pnl'])],
        'Std_Dev':[np.std(position_df['pnl'])],
        'Skew':[np.float(skew(position_df['pnl']))],
        'Kurtosis':[np.float(kurtosis(position_df['pnl']))],
    }

    return pd.DataFrame(performance_dict)

