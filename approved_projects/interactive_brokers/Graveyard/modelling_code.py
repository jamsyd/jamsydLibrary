import numpy as np 
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

import math
from itertools import accumulate
from scipy.stats import skew, kurtosis

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
\
def apply_transformation(timeseries,transformation):

    """
        Description: 

        :param timeseries: The timeseries
        :param transformation: The transformation to apply to the data
    """

    transformed_timeseries = None

    if transformation == 'log_diff':
        transformed_timeseries = 100*(np.exp(np.log(timeseries).diff(1)) - 1)

    if transformation == 'first_diff':
        transformed_timeseries = timeseries.diff(1)

    return transformed_timeseries

def log_difference(forecast_horizon, historical_series, forecast_order, product, moving_average):

    """
    
        Description: 

        :param forecast_horizon:
        :param historical_series:
        :param forecast_order:
        :param product: 
    
    """

    all_forecasts = []

    model_metadata = {

        "model_id":[],
        "aic":[],
        "order":[],
        "mae":[],
        "mse":[],
        "skew":[],
        "kurtosis":[]

    }

    # Read in the data
    # ts_df = pd.read_csv(rf'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\BATS_{product}, 1D.csv',index_col='time',parse_dates=True)
    ts_df = pd.read_csv(rf'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\ICEUS_DLY_CT1!, 1D.csv',index_col='time',parse_dates=True)
    # Perform initial transformations
    ts_df['transformation'] = apply_transformation(timeseries = ts_df['close'], transformation = "log_diff")
    ts_df['ma']             = ts_df['close'].rolling(window=moving_average).mean()
    ts_df['ma_diff']        = ts_df['ma'].diff(1)
    ts_df['pnl']            = ts_df['close'].diff(1)
    ts_df                   = ts_df.dropna(axis=0)

    # Assuming ts_df is already defined and has a DateTime index
    ts_df.index = pd.to_datetime(ts_df.index)

    # Infer the frequency of the DateTime index
    inferred_freq = pd.infer_freq(ts_df.index)
    if inferred_freq:
        ts_df = ts_df.asfreq(inferred_freq)

    i = 0
    while i < len(ts_df):

        # Ensure there are enough observations for historical_series
        if i + historical_series >= len(ts_df):
            break

        end = min(i + forecast_horizon, len(ts_df))

        # Corrected the slicing of the DataFrame
        mod = sm.tsa.arima.ARIMA(ts_df['transformation'].iloc[i:i + historical_series], order=forecast_order)
        res = mod.fit()

        # Forecast the next 5 days
        forecast = res.forecast(steps=forecast_horizon)

        # Taking previous day
        last_close        = ts_df['close'].iloc[i + historical_series - 1]
        forecasts         = ((1 + forecast / 100) * last_close).tolist()
        forecast_dates    = ts_df.index[i + historical_series:i + historical_series + forecast_horizon].tolist()
        close_actual      = ts_df['close'].iloc[i + historical_series:i + historical_series + forecast_horizon].tolist()
        model_id          = product + '_' + str(i) + "_" + str(forecast_order)

        # store model metadata
        model_metadata['model_id'].append(model_id)
        model_metadata['aic'].append(res.aic)
        model_metadata['order'].append(str(forecast_order))
        model_metadata['mae'].append(np.mean(np.abs(res.resid)))
        model_metadata['mse'].append(np.mean(np.square(res.resid)))
        model_metadata['skew'].append(skew(res.resid))
        model_metadata['kurtosis'].append(kurtosis(res.resid))

        # Information to store in list
        forecast_position = [i for i in range(1,forecast_horizon + 1)]
        model_id_lst      = forecast_horizon*[model_id]
        last_price        = forecast_horizon*[last_close]
        std_dev           = forecast_horizon*[np.std(ts_df['close'].iloc[i:i + historical_series].diff(1))]
        mean              = forecast_horizon*[np.mean(ts_df['close'].iloc[i:i + historical_series].diff(1))]
        ma_diff           = forecast_horizon*[ts_df['ma_diff'].iloc[i + historical_series - 1]]

        # Append the forecasts and corresponding dates to all_forecasts
        all_forecasts.extend(list(zip(forecast_dates, forecasts, close_actual, forecast_position, model_id_lst, last_price, std_dev, mean, ma_diff)))
        i += forecast_horizon 

        if i > 100:
            break


    # Saving the forecast dataframe
    forecast_df          = pd.DataFrame(all_forecasts, columns=['Date', 'Forecast', 'Close', 'Forecast_Position', 'model_id','last_price', 'standard_dev', 'mean','ma_diff'])
    forecast_df['error'] = forecast_df['Close'] - forecast_df['Forecast']

    # Store model metadata
    pd.DataFrame(model_metadata).to_csv(rf'model_metadata_log_first_diff_{product}_{forecast_order}_{historical_series}_{forecast_horizon}.csv')
    
    return forecast_df


def first_difference(forecast_horizon, historical_series, forecast_order, product, moving_average):

    """
    
        Description: 

        :param forecast_horizon:
        :param historical_series:
        :param forecast_order:
        :param product: 
        :param moving_average:
    
    """

    all_forecasts = []

    model_metadata = {

        "model_id":[],
        "aic":[],
        "order":[],
        "mae":[],
        "mse":[],
        "skew":[],
        "kurtosis":[]

    }

    # Read in the data
    # ts_df = pd.read_csv(rf'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\BATS_{product}, 1D.csv',index_col='time',parse_dates=True)
    ts_df = pd.read_csv(rf'C:\Users\james\OneDrive\Documents\GitHub\jamsydLibrary\approved_projects\interactive_brokers\Data\ICEUS_DLY_CT1!, 1D.csv',index_col='time',parse_dates=True)

    # Perform initial transformations
    ts_df['transformation'] = apply_transformation(timeseries = ts_df['close'], transformation = "first_diff")
    ts_df['ma']             = ts_df['close'].rolling(window=moving_average).mean()
    ts_df['ma_diff']        = ts_df['ma'].diff(1)
    ts_df['pnl']            = ts_df['close'].diff(1)
    ts_df                   = ts_df.dropna(axis=0)

    # Assuming ts_df is already defined and has a DateTime index
    ts_df.index = pd.to_datetime(ts_df.index)

    # Infer the frequency of the DateTime index
    inferred_freq = pd.infer_freq(ts_df.index)
    if inferred_freq:
        ts_df = ts_df.asfreq(inferred_freq)

    i = 0
    while i < len(ts_df):

        # Ensure there are enough observations for historical_series
        if i + historical_series >= len(ts_df):
            break

        end = min(i + forecast_horizon, len(ts_df))

        # Corrected the slicing of the DataFrame
        mod = sm.tsa.arima.ARIMA(ts_df['transformation'].iloc[i:i + historical_series], order=forecast_order)
        res = mod.fit()

        # Forecast the next 5 days
        forecast = res.forecast(steps=forecast_horizon)

        # Taking previous day
        last_close = ts_df['close'].iloc[i + historical_series - 1]
        forecasts  = pd.DataFrame(forecast)
        forecasts  = forecast.cumsum()
        forecasts  = forecasts.reset_index()

        print(forecasts['predicted_mean'].iloc[0])

        j = 0
        while j < forecast_horizon:
            print(j,forecasts.index)
            forecasts['predicted_mean'].iloc[j] = forecasts['predicted_mean'].iloc[j] + last_close
            j+=1

        forecasts = forecasts['predicted_mean'].to_list()

        # Calculating forecasts
        forecast_dates    = ts_df.index[i + historical_series:i + historical_series + forecast_horizon].tolist()
        close_actual      = ts_df['close'].iloc[i + historical_series:i + historical_series + forecast_horizon].tolist()
        pnl               = ts_df['pnl'].iloc[i + historical_series:i + historical_series + forecast_horizon].tolist()
        model_id          = product + '_' + str(i) + "_" + str(forecast_order)

        # store model metadata
        model_metadata['model_id'].append(model_id)
        model_metadata['aic'].append(res.aic)
        model_metadata['order'].append(str(forecast_order))
        model_metadata['mae'].append(np.mean(np.abs(res.resid)))
        model_metadata['mse'].append(np.mean(np.square(res.resid)))
        model_metadata['skew'].append(skew(res.resid))
        model_metadata['kurtosis'].append(kurtosis(res.resid))

        # Information to store in list
        forecast_position = [i for i in range(1, forecast_horizon + 1)]
        model_id_lst      = forecast_horizon*[model_id]
        last_price        = forecast_horizon*[last_close]
        std_dev           = forecast_horizon*[np.std(ts_df['close'].iloc[i:i + historical_series].diff(1))]
        mean              = forecast_horizon*[np.mean(ts_df['close'].iloc[i:i + historical_series].diff(1))]
        ma_diff           = forecast_horizon*[ts_df['ma_diff'].iloc[i + historical_series - 1]]

        # Append the forecasts and corresponding dates to all_forecasts
        all_forecasts.extend(list(zip(forecast_dates, forecasts, close_actual, forecast_position, model_id_lst, last_price, std_dev, mean, ma_diff, pnl)))
        i += forecast_horizon 

    # Saving the forecast dataframe
    forecast_df          = pd.DataFrame(all_forecasts, columns=['Date', 'Forecast', 'Close', 'Forecast_Position', 'model_id','last_price', 'standard_dev', 'mean','ma_diff','pnl'])
    forecast_df          = forecast_df.dropna(axis=0)
    forecast_df['error'] = forecast_df['Close'] - forecast_df['Forecast']

    # Store model metadata
    pd.DataFrame(model_metadata).to_csv(rf'model_metadata_first_diff_{product}_{forecast_order}_{historical_series}_{forecast_horizon}.csv')
    
    return forecast_df


modelling_payload = {"forecast_horizon": 3,
                     "moving_average": 18,
                     "historical_series": 300,
                     "forecast_order": (3, 0, 0),
                     "product": 'EWA'}

# res  = log_difference(**modelling_payload)
res2 = first_difference(**modelling_payload)
res2.to_csv(r'test.csv')

# for i in range(1,forecast_horizon+1):
#     print(f"Mean {i}",forecast_df[forecast_df['Forecast_Position'] == i]['error'].mean())
#     print(f"Std {i}",forecast_df[forecast_df['Forecast_Position'] == i]['error'].std()/np.std(forecast_df['Close'].dropna(axis=0).diff(i)))
#     print(f"Skew {i}",skew(forecast_df[forecast_df['Forecast_Position'] == i]['error']))
#     print(f"Kurtosis {i}",kurtosis(forecast_df[forecast_df['Forecast_Position'] == i]['error']))



# pnl = []
# for model_id in forecast_df['model_id'].unique():

#     forecast_move     = forecast_df[forecast_df['model_id'] == model_id]['Forecast'].iloc[-1] - forecast_df[forecast_df['model_id'] == model_id]['last_price'].iloc[0]
#     forecast_move_std = forecast_df[forecast_df['model_id'] == model_id]['standard_dev'].iloc[0]
    
#     print(forecast_move)

#     if forecast_move < 0:
#         if forecast_df[forecast_df['model_id'] == model_id]['ma_diff'].iloc[-1] < 0:
#             pnl.extend(-1*forecast_df[forecast_df['model_id'] == model_id]['PnL'].to_list())

#     if forecast_move > 0:
#         if forecast_df[forecast_df['model_id'] == model_id]['ma_diff'].iloc[-1] > 0:
#             pnl.extend(forecast_df[forecast_df['model_id'] == model_id]['PnL'].to_list())

# # Replace NaN values with zero
# pnl    = [0 if math.isnan(x) else x for x in pnl]
# vector = list(accumulate(pnl))
    