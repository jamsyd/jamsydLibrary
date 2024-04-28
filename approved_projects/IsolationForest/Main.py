import numpy as np 
import pandas as pd 

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

# reading in initial dataframe & sorting it
init_df = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\jamsydLibrary\data\market_data\daily\stocks\BATS_SPY, 1D.csv',index_col='time',parse_dates=True)
vix_df  = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\jamsydLibrary\data\market_data\daily\stocks\CBOE_VX1!, 1D.csv',index_col='time',parse_dates=True)

init_df = init_df.add_suffix('_SPY')  # Adding '_SPY' to columns from init_df
vix_df  = vix_df.add_suffix('_VX1')

init_df = init_df.sort_values(by='time', ascending=True)

merged_df = pd.merge(init_df, vix_df, left_index=True, right_index=True, how='inner')

# Creating our dataframe for input
diff_df = np.exp(np.log(init_df[['open_SPY','high_SPY','low_SPY','close_SPY','open_VX1','high_VX1','low_VX1','close_VX1']]).diff(1))
diff_df = diff_df.dropna(axis=0)

# Setting PnL Vector
init_df['pnl']        = init_df['close_SPY'].diff(1).shift(-1)

# Only down days
diff_df = diff_df[diff_df['close_SPY'].diff(1) < 0]

X_train = diff_df.iloc[0:700]
X_test  = diff_df.iloc[700:]

print(len(diff_df),len(X_train),len(X_test))

# Train our model
model = IsolationForest(contamination=0.1)
model.fit(X_train)

y_pred = model.predict(X_test)
y_pred[y_pred == -1] = 0

print(len(y_pred))

# Generate Predictions
predictions = pd.DataFrame(y_pred, index=X_test.index)
merge_df    = pd.merge(init_df[['close','pnl']],predictions, on='time')
print(len(predictions),len(merge_df))

merge_df[0]           = 1 - merge_df[0]
merge_df['final_pnl'] = merge_df['pnl']*merge_df[0]

merge_df.to_csv(r'pnl.csv')