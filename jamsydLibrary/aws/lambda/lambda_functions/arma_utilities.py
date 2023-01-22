import os
import sys

import numpy as np
import pandas as pd

from datetime import datetime
from statsmodels.tsa.arima_model import ARMA


diff_df = pd.read_csv(r'')

start_train = ''
end_train   = ''

mod = ARMA(diff_df[(diff_df['asofdate'] > start_train) and (diff_df['asofdate'] > end_train)]['close'])
res = mod.fit()

# converting log forecast
fcast = res.forecast(self.forecastHorizon)

                    cacheForecasts['pointForecast'].append(np.exp(fcast[0])*self.dataframe[self.column][self.trainDFLength+i])
                    cacheForecasts[self.column].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])