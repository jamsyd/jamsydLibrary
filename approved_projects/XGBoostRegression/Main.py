import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 

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

df['firstlag']  = df[['close']].shift(-1)
df['secondlag'] = df[['close']].shift(-2)
df['thirdlag'] = df[['close']].shift(-3)
df['fourthlag'] = df[['close']].shift(-4)
df['fifthlag'] = df[['close']].shift(-5)

df = df.dropna(axis=0)
#######################################################################################################
# Vix
#######################################################################################################
df_vix = np.exp(np.log(df_vix[['close']]).diff(1))

df_vix['firstlag_vix']  = df_vix[['close']].shift(-1)
df_vix['secondlag_vix'] = df_vix[['close']].shift(-2)
df_vix['thirdlag_vix'] = df_vix[['close']].shift(-3)
df_vix['fourthlag_vix'] = df_vix[['close']].shift(-4)
df_vix['fifthlag_vix'] = df_vix[['close']].shift(-5)

df_vix = df_vix[['firstlag_vix','secondlag_vix','thirdlag_vix','fourthlag_vix','fifthlag_vix']]

df_vix = df_vix.dropna(axis=0)

df_vix.index = df_vix.index.date

#######################################################################################################
# Vix
#######################################################################################################
df_vix_dif = df_vix_dif[['close']].diff(1)

df_vix_dif['firstlag_vix_dif']  = df_vix_dif[['close']].shift(-1)
df_vix_dif['secondlag_vix_dif'] = df_vix_dif[['close']].shift(-2)
df_vix_dif['thirdlag_vix_dif'] = df_vix_dif[['close']].shift(-3)
df_vix_dif['fourthlag_vix_dif'] = df_vix_dif[['close']].shift(-4)
df_vix_dif['fifthlag_vix_dif'] = df_vix_dif[['close']].shift(-5)

df_vix_dif = df_vix_dif[['firstlag_vix_dif','secondlag_vix_dif','thirdlag_vix_dif','fourthlag_vix_dif','fifthlag_vix_dif']]

df_vix_dif = df_vix_dif.dropna(axis=0)

df_vix_dif.index = df_vix_dif.index.date








df = pd.concat([df,df_vix,df_vix_dif],axis=1)
df = df.dropna(axis=0)

print(df)

forecast_df_lst = []
i = 0
while i < 500: #len(df) - 5:

    # Set train and test data
    X_train = df[['firstlag','secondlag','thirdlag','fourthlag','fifthlag','firstlag_vix','secondlag_vix','thirdlag_vix','fourthlag_vix','fifthlag_vix','firstlag_vix_dif','secondlag_vix_dif','thirdlag_vix_dif','fourthlag_vix_dif','fifthlag_vix_dif']].iloc[i:i+train_size].dropna(axis=0) #'thirdlag','fourthlag','fifthlag'
    X_test  = df[['firstlag','secondlag','thirdlag','fourthlag','fifthlag','firstlag_vix','secondlag_vix','thirdlag_vix','fourthlag_vix','fifthlag_vix','firstlag_vix_dif','secondlag_vix_dif','thirdlag_vix_dif','fourthlag_vix_dif','fifthlag_vix_dif']].iloc[i+train_size:i+train_size+forecast_horizon].dropna(axis=0)

    y_train = df[['close']].iloc[i:i+train_size].dropna(axis=0)
    y_test  = df[['close']].iloc[i+train_size:i+train_size+forecast_horizon].dropna(axis=0)

    # # Assuming you have already defined and fitted your model:
    # model = XGBRegressor(n_estimators=15, learning_rate=10, objective='reg:squarederror')
    # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees in the forest
        max_depth=12,  # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,  # Minimum number of samples required to be at a leaf node
        random_state=42  # For reproducibility
    )

    # Fit the model (your fitting code remains the same)
    model.fit(
        X_train, y_train  # Assuming y_train is reshaped if necessary
    )

    forecasts = []

    # Forecast
    # latest_input = X_test.iloc[-1].values[-lag:]
    latest_input = X_test.iloc[-1].values
    # forecast_values = forecast(model, latest_input, days=forecast_horizon)
    forecast_values = forecast(model, latest_input, days=forecast_horizon)
    forecasts.append(forecast_values)

    # Append forecasts
    forecasts.append(y_test.index.T)

    # Creating dataframe
    forecast_df = pd.DataFrame(forecasts).T
    forecast_df = forecast_df.set_index(1)
    forecast_df.index.name = 'time'
    forecast_df.rename(columns={0: 'forecast'}, inplace=True)
    forecast_df['actuals'] = y_test.values
    forecast_df['forecast_day'] = [i for i in range(1,6)]
    forecast_df['origin'] = X_train.index[-1]
    
    forecast_df_lst.append(forecast_df)


    print(forecast_df)

    i+=forecast_horizon


pd.concat(forecast_df_lst).to_csv(r'fcast.csv')