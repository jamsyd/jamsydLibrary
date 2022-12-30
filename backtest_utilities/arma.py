import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd

from statsmodels.tsa.arima_model import ARMA

class train_arma():
    
    def __init__(self, event):

        # Read in dataframe
        self.dataframe       = pd.read_csv(event['dataframe'],parse_dates=True,index_col='time')

        # Training information
        self.forecastHorizon = event['forecastHorizon']
        self.trainDFLength   = event['trainDFLength']
        self.order           = event['order']
        self.log_diff        = event['log_diff']

        # number of mdoels to be trained
        self.num_models      = event['num_models']

        # store l of forecast
        self.forecastList    = []

    def arma_model(self):

        cacheForecasts = {

            'asofdate':[],
            'pointForecast':[],
            'close':[],
            'forecastday':[],

        }

        cacheMetadata = {

            'asofdate':[],
            'aic':[],
            'bic':[],
            'hqic':[],
            'mae':[],
            'mse':[],
            'rsquare':[],
            'forecastHorizon':[],
            'log_diff':[],
            'trainDFLength':[],
            'num_models':[],
            'order_p':[],
            'order_q':[],
            'time':[],

        }

        for param in range(1,1+self.order[0]):
            cacheMetadata[f'ar.L{param}.close'] = []

        for param in range(1,1+self.order[1]):
            cacheMetadata[f'ma.L{param}.close'] = []

        if self.log_diff:
            log_diff   = np.log(self.dataframe['close']).diff(1).dropna(axis=0)
    
            for j in range(1,self.forecastHorizon+1):
                self.forecastList.append(j)

            i = 0
            while i < self.num_models:

                cacheForecasts['forecastday'].append(self.forecastList)

                try:
                    start = datetime.now()

                    # training the model
                    mod = ARMA(log_diff[i:self.trainDFLength+i], order=self.order)
                    res = mod.fit()
                    
                    # converting log forecast
                    log_fcast     = res.forecast(self.forecastHorizon)

                    # Calculating forecasts
                    cacheForecasts['pointForecast'].append(np.exp(log_fcast[0])*self.dataframe['close'][self.trainDFLength+i])

                    # store model metadata
                    cacheMetadata['asofdate'].append(self.dataframe.index[i])
                    cacheMetadata['aic'].append(res.aic)
                    cacheMetadata['bic'].append(res.bic)
                    cacheMetadata['hqic'].append(res.hqic)
                    # cacheMetadata['bse'].append(res.bse)
                    cacheMetadata['mae'].append(np.mean(np.abs(res.resid)))
                    cacheMetadata['mse'].append(np.mean(np.square(res.resid)))
                    cacheMetadata['rsquare'].append(np.mean(np.square(res.resid))/np.var(np.exp(log_diff[i:self.trainDFLength+i])))

                    # event_data
                    cacheMetadata['forecastHorizon'].append(self.forecastHorizon)
                    cacheMetadata['log_diff'].append(self.log_diff)
                    cacheMetadata['trainDFLength'].append(self.trainDFLength)
                    cacheMetadata['num_models'].append(self.num_models)
                    cacheMetadata['order_p'].append(self.order[0])
                    cacheMetadata['order_q'].append(self.order[1])

                    for param in range(1,1+self.order[0]):
                        cacheMetadata[f'ar.L{param}.close'].append(res.bse[f'ar.L{param}.close'])

                    for param in range(1,1+self.order[1]):
                        cacheMetadata[f'ma.L{param}.close'].append(res.bse[f'ma.L{param}.close'])

                    cacheMetadata['time'].append(datetime.now() - start)

                except:
                    # In case model does not converge 
                    cacheForecasts['pointForecast'].append(self.dataframe['close'][self.trainDFLength+i:\
                                                        self.trainDFLength+self.forecastHorizon+i])

                i+=self.forecastHorizon

            # Fixing data formats
            cacheForecasts['pointForecast'] = np.concatenate(cacheForecasts['pointForecast'])
            cacheForecasts['asofdate']      = np.array(self.dataframe['close'][self.trainDFLength+self.forecastHorizon:\
                                                    self.trainDFLength+self.forecastHorizon+\
                                                    len(cacheForecasts['pointForecast'])].index)
            cacheForecasts['close']         = np.array(self.dataframe['close'][self.trainDFLength+self.forecastHorizon:\
                                                    self.trainDFLength+self.forecastHorizon+len(cacheForecasts['pointForecast'])])
            cacheForecasts['forecastday']   = np.concatenate(cacheForecasts['forecastday'])

            pd.DataFrame(cacheForecasts).to_csv('forecasts.csv')
            pd.DataFrame(cacheMetadata).to_csv('metadata.csv')

        if not self.log_diff:
            diff   = self.dataframe['close'].diff(1).dropna(axis=0)
    
            for i in range(1,self.forecastHorizon+1):
                self.forecastList.append(i)

            i = 0
            while i < self.num_models:

                cacheForecasts['forecastday'].append(self.forecastList)

                try:
                    # training the model
                    mod = ARMA(diff[i:self.trainDFLength+i], order=self.order)
                    res = mod.fit()
                    
                    # converting log forecast
                    fcast     = res.forecast(self.forecastHorizon)

                    # storing forecasts
                    cacheForecasts['pointForecast'].append(fcast[0]*self.dataframe['close'][self.trainDFLength+i])

                    # store model metadata
                    # store model metadata
                    cacheMetadata['asofdate'].append(self.dataframe.index[i])
                    cacheMetadata['aic'].append(res.aic)
                    cacheMetadata['bic'].append(res.bic)
                    cacheMetadata['hqic'].append(res.hqic)
                    # cacheMetadata['bse'].append(res.bse)
                    cacheMetadata['mae'].append(np.mean(np.abs(res.resid)))
                    cacheMetadata['mse'].append(np.mean(np.square(res.resid)))
                    cacheMetadata['rsquare'].append(np.mean(np.square(res.resid))/np.var(np.exp(log_diff[i:self.trainDFLength+i])))
                    
                    # event_data
                    cacheMetadata['forecastHorizon'].append(self.forecastHorizon)
                    cacheMetadata['log_diff'].append(self.log_diff)
                    cacheMetadata['trainDFLength'].append(self.trainDFLength)
                    cacheMetadata['num_models'].append(self.num_models)
                    cacheMetadata['order_p'].append(self.order[0])
                    cacheMetadata['order_q'].append(self.order[1])

                    for param in range(1,1+self.order[0]):
                        cacheMetadata[f'ar.L{param}.close'].append(res.bse[f'ar.L{param}.close'])

                    for param in range(1,1+self.order[1]):
                        cacheMetadata[f'ma.L{param}.close'].append(res.bse[f'ma.L{param}.close'])

                except:
                    # In case model does not converge 
                    cacheForecasts['pointForecast'].append(self.dataframe['close'][self.trainDFLength+i:\
                                                        self.trainDFLength+self.forecastHorizon+i])

                i+=self.forecastHorizon

            # Fixing data formats
            cacheForecasts['pointForecast'] = np.concatenate(cacheForecasts['pointForecast'])
            cacheForecasts['asofdate']      = np.array(self.dataframe['close'][self.trainDFLength+self.forecastHorizon:\
                                                    self.trainDFLength+self.forecastHorizon+\
                                                    len(cacheForecasts['pointForecast'])].index)
            cacheForecasts['close']         = np.array(self.dataframe['close'][self.trainDFLength+self.forecastHorizon:\
                                                    self.trainDFLength+self.forecastHorizon+len(cacheForecasts['pointForecast'])])
            cacheForecasts['forecastday']   = np.concatenate(cacheForecasts['forecastday'])

            pd.DataFrame(cacheForecasts).to_csv('forecasts.csv')
            pd.DataFrame(cacheMetadata).to_csv('metadata.csv')


class model_analysis(train_arma):

    def __init__(self,train_arma):
        self.model = train_arma

    def forecast_error(self):

        cacheForecastError = {

            'asofdate':self.model.asofdate,
            'forecastError':self.model.close - self.model.pointForecast,

        }

        return pd.DataFrame(cacheForecastError)

model_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\Corn\Data\CBOT_DL_ZC1!, 1D.csv',
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(2,2),
    'num_models':12000,
    'log_diff':True,

}

model = train_arma(model_event)

model.arma_model()