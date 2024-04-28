import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings(action='ignore', category=FutureWarning)

def forecast(model, input_data, days):
    """
    Generate out-of-sample forecasts.
    """
    
    forecast = []
    current_input = input_data.copy()

    for _ in range(days):
        next_day_pred = model.predict(current_input.reshape(1, -1))[0]
        forecast.append(next_day_pred)
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_day_pred

    return forecast


############################################################################################################
# Reading in the Data
############################################################################################################
df = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\jamsydLibrary\data\market_data\daily\stocks\CME_MINI_DL_ES1!, 1D (1).csv', parse_dates=True, index_col='time')
df = df.sort_index(ascending=True)  # Ensure data is sorted by time

df_vix = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\jamsydLibrary\data\market_data\daily\stocks\CBOE_DLY_VX1!, 1D.csv', parse_dates=True, index_col='time')
df_vix = df_vix.sort_index(ascending=True)  # Ensure data is sorted by time

df_vix_dif = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\jamsydLibrary\data\market_data\daily\stocks\CBOE_VX2!-CBOE_VX1!, 1D.csv', parse_dates=True, index_col='time')
df_vix_dif = df_vix_dif.sort_index(ascending=True)  # Ensure data is sorted by time

# Parameters
train_size = 1260  
forecast_horizon = 5
lag = 5

df = np.exp(np.log(df[['close']]).diff(1))
df = df[['close']]

df = df.dropna(axis=0)
#######################################################################################################
# Vix
#######################################################################################################
df_vix = np.exp(np.log(df_vix[['close']]).diff(1))

df_vix = df_vix[['close']]

df_vix = df_vix.dropna(axis=0)

df_vix.index = df_vix.index.date

#######################################################################################################
# Vix
#######################################################################################################
df_vix_dif = df_vix_dif[['close']].diff(1)

df_vix_dif = df_vix_dif[['close']]

df_vix_dif = df_vix_dif.dropna(axis=0)

df_vix_dif.index = df_vix_dif.index.date

df = pd.concat([df,df_vix,df_vix_dif],axis=1)
df = df.dropna(axis=0)

# df.index = pd.DatetimeIndex(df.index).asfreq('D')
df = df.sort_index()

# print(df)
# forecast_df_lst = []
# i = 0
# while i < 500:  # Adjusted as needed for your example

#     # Set train and test data
#     data = df.iloc[i:i+train_size+forecast_horizon]
    
#     # Fit the VAR model
#     model = VAR(data)
#     model_fitted = model.fit(lag)


#     # Forecast
#     forecast = model_fitted.forecast(steps=forecast_horizon)
#     print(forecast)

# Forecasting loop
forecast_df_lst = []  # List to store forecasts
for i in range(0, len(df) - train_size - forecast_horizon + 1, forecast_horizon):
    # Set train data
    train_data = df.iloc[i:i+train_size]
    
    # Fit the VAR model
    model = VAR(train_data)
    model_fitted = model.fit(lag)
    
    # Forecast
    last_obs = train_data.values[-lag:]
    fcast = model_fitted.forecast(y=last_obs, steps=forecast_horizon)
    forecast_df_lst.append(fcast)  # Store forecast
    print(fcast)

#     # Prepare forecast dataframe
#     forecast_df = pd.DataFrame(forecast_values, columns=data.columns)
    
#     # Assuming you want to compare forecasted 'close' values to actuals
#     actuals = df['close'].iloc[i+train_size:i+train_size+forecast_horizon].values
#     forecast_df['actuals'] = actuals
#     forecast_df['forecast_day'] = [i for i in range(1, forecast_horizon+1)]
#     forecast_df['origin'] = data.index[-1]
    
#     forecast_df_lst.append(forecast_df)

#     i += forecast_horizon

# # Assuming you want to concatenate all forecast_df into one and save it
# final_forecast_df = pd.concat(forecast_df_lst)
# final_forecast_df.to_csv(r'fcast.csv')