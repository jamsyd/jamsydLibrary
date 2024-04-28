import warnings
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings(action='ignore', category=FutureWarning)

def prepare_data(input_data, n_features):
    """
    Convert input data into a 3D array format required for LSTM.
    """
    data = input_data.values
    X, y = [], []
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_features
        if end_ix > len(data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def forecast(model, input_data, n_features, days=5):
    """
    Generate out-of-sample forecasts.
    """
    forecast = []
    current_input = input_data.reshape((1, n_features, 1))
    
    for _ in range(days):
        next_day_pred = model.predict(current_input)[0][0]
        forecast.append(next_day_pred)
        current_input = np.roll(current_input, -1)
        current_input[0, -1, 0] = next_day_pred

    return forecast

# Reading in the Data
df = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\jamsydLibrary\data\market_data\daily\stocks\BATS_SPY, 1D.csv', parse_dates=True, index_col='time')
df = df.sort_index(ascending=True)  # Ensure data is sorted by time
df = np.log(df[['close']].pct_change() + 1).dropna()

# Parameters
n_features = 5  # Number of past days you want to use to predict the future
forecast_horizon = 5

# Preparing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])
X, y = prepare_data(pd.DataFrame(scaled_data), n_features)

# Reshaping X for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Splitting the dataset into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_features, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

# Forecasting
latest_input = scaled_data[-n_features:]
forecast_values = forecast(model, latest_input, n_features, days=forecast_horizon)
forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))

print(forecast_values)
