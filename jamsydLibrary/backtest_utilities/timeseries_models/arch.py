import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

import arch
from arch import arch_model

def ROC(df, n):  
    M = df.diff(n - 1)  
    N = df.shift(n - 1)  
    ROC = pd.Series(((M / N) * 100), name = 'ROC_' + str(n))   
    return ROC

class train_arch(arch_event):

    def __init__(self, event):

        # Read in dataframe
        self.dataframe       = pd.read_csv(event['dataframe'],parse_dates=True,index_col='time')

        # Training information
        self.forecastHorizon = event['forecastHorizon']
        self.trainDFLength   = event['trainDFLength']
        self.order           = event['order']
        self.log_diff        = event['log_diff']
        self.product         = event['product']

        # number of mdoels to be trained
        self.num_models      = event['num_models']

        # store l of forecast
        self.forecastList    = []

        # self.returns
        self.returns = ROC(df=self.dataframe[self.column],n=2)

    def arch(self):
        
        returns = ROC(df=self.dataframe=self.dataframe[self.column],n=2)
        # returns = returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        returns = returns.dropna()
        am = arch_model(returns, p=1, o=1, q=1, power=1.0, dist="StudentsT")
        
        res = am.fit(update_freq=5)

        fixed_res = am.fix

    return pd.concat([res.conditional_volatility, fixed_res.conditional_volatility], 1)



arma_model_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\data\daily\orange_juice\ICEUS_DLY_OJ1!, 1D.csv',
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(2,2),
    'num_models':12000,
    'log_diff':True,
    'product':'orange_juice',
    'column':'close'

}
