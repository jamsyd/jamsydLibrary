import os
import sys

import numpy as np
import pandas as pd

from datetime import datetime
from statsmodels.tsa.arima_model import ARMA

class trainResidualSeasonality:
    
    def __init__(self, event):

        # Read in dataframe
        self.dataframe       = event['dataframe'] # pd.read_csv(event['dataframe'],parse_dates=True,index_col='time')

        # Training information
        self.forecastHorizon = event['forecastHorizon']
        self.trainDFLength   = event['trainDFLength']
        self.order           = event['order']
        self.diff            = event['diff']
        self.product         = event['product']
        self.column          = event['column']
        self.b_adjust        = event['b_adjust']

        # number of mdoels to be trained
        self.num_models      = event['num_models']

        # store l of forecast
        self.forecastList    = []
        self.event           = event

    def arma_model(self):

        cacheForecasts = {

            'asofdate':[],
            'residualForecast':[],
            self.column:[],
            'forecastday':[],
            'product_name':[],
            # 'backtestMethod':[],

        }

        cacheMetadata = {

            'asofdate':[],
            'aic':[],
            'bic':[],
            'hqic':[],
            'mae':[],
            'mse':[],
            'forecastHorizon':[],
            'diff':[],
            'trainDFLength':[],
            'num_models':[],
            'order_p':[],
            'order_q':[],
            'time':[],
            'product_name':[],
            # 'backtestMethod':[],

        }

        for param in range(1,1+self.order[0]):
            cacheMetadata[f'ar.L{param}.{self.column}'] = []

        for param in range(1,1+self.order[1]):
            cacheMetadata[f'ma.L{param}.{self.column}'] = []

        if self.diff:
            diff   = np.log(self.dataframe[self.column]).diff(1).dropna(axis=0)
    
            for j in range(1,self.forecastHorizon+1):
                self.forecastList.append(j)

            i = 0
            while i < self.num_models:

   
                try:
                    start = datetime.now()

                    # training the model
                    mod = ARMA(diff[i:self.trainDFLength+i], order=self.order)
                    res = mod.fit()
                    
                    # converting log forecast
                    fcast = res.forecast(self.forecastHorizon)

                    # Calculating forecasts
                    cacheForecasts['residualForecast'].append(np.exp(fcast[0])*self.dataframe[self.column][self.trainDFLength+i])
                    cacheForecasts[self.column].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])

                    # store model metadata
                    cacheMetadata['asofdate'].append(self.dataframe.index[self.trainDFLength+i])
                    cacheMetadata['aic'].append(res.aic)
                    cacheMetadata['bic'].append(res.bic)
                    cacheMetadata['hqic'].append(res.hqic)
                    cacheMetadata['mae'].append(np.mean(np.abs(res.resid)))
                    cacheMetadata['mse'].append(np.mean(np.square(res.resid)))

                    # event_data
                    cacheMetadata['forecastHorizon'].append(self.forecastHorizon)
                    cacheMetadata['diff'].append(self.diff)
                    cacheMetadata['trainDFLength'].append(self.trainDFLength)
                    cacheMetadata['num_models'].append(self.num_models)
                    cacheMetadata['order_p'].append(self.order[0])
                    cacheMetadata['order_q'].append(self.order[1])
                    cacheMetadata['product_name'].append(self.product)

                    for param in range(1,1+self.order[0]):
                        cacheMetadata[f'ar.L{param}.{self.column}'].append(res.bse[f'ar.L{param}.{self.column}'])

                    for param in range(1,1+self.order[1]):
                        cacheMetadata[f'ma.L{param}.{self.column}'].append(res.bse[f'ma.L{param}.{self.column}'])

                    cacheMetadata['time'].append(datetime.now() - start)
                    # cacheMetadata['backtestMethod'].append('multiplicative_seasonal_residual')

                except:
                    cacheForecasts['residualForecast'].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])

                i+=self.forecastHorizon

            # Fixing data formats
            cacheForecasts['residualForecast'] = np.concatenate(cacheForecasts['residualForecast'])
            cacheForecasts['forecastday']      = np.array(self.forecastList*(int(len(cacheForecasts['residualForecast'])/int(self.forecastHorizon))))
            cacheForecasts[self.column]        = np.array(self.dataframe[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['residualForecast'])])
            cacheForecasts['asofdate']         = np.array(self.dataframe[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['residualForecast'])].index)
            cacheForecasts['product_name']     = np.array([self.product]*int(len(cacheForecasts['residualForecast'])))
            # cacheForecasts['backtestMethod']   = len(cacheForecasts['product_name'])*['multiplicative_seasonal_residual']

            # pd.DataFrame(cacheForecasts).to_csv(f'forecasts_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}.csv')
            pd.DataFrame(cacheMetadata).to_csv(f'metadata_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}.csv')
        
        return (pd.DataFrame(cacheForecasts),pd.DataFrame(cacheMetadata))
