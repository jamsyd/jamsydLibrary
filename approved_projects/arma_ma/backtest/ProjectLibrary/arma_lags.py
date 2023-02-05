import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

arma_df = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\approved_strategies\corn_arma_ma\data\CBOT_DL_ZC1!, 1D.csv',index_col='time', parse_dates=True)
log_diff = np.log(arma_df['close']).diff(1).dropna(axis=0)

plot_acf(log_diff, alpha=1, lags=5).savefig("corn_acf.jpg")
plot_pacf(log_diff, alpha=1, lags=5).savefig("corn_pacf.jpg")