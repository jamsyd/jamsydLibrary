import os
import sys

import numpy as np
import pandas as pd

from datetime import datetime
from statsmodels.tsa.arima_model import ARMA

class train_arma:
    
    def __init__(self, event):

        # Read in dataframe
        self.dataframe       = pd.read_csv(event['dataframe'],parse_dates=True,index_col='time')

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

    def arma_model(self):

        cacheForecasts = {

            'asofdate':[],
            'pointForecast':[],
            self.column:[],
            'forecastday':[],
            'product_name':[],
            'MA_20':[],
            'MA_50':[],
            'MA_200':[],
            'MA_diff_20':[],
            'MA_diff_50':[],
            'MA_diff_200':[],

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
            'diff':[],
            'trainDFLength':[],
            'num_models':[],
            'order_p':[],
            'order_q':[],
            'time':[],
            'product_name':[],

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
                    cacheForecasts['pointForecast'].append(np.exp(fcast[0])*self.dataframe[self.column][self.trainDFLength+i])
                    cacheForecasts[self.column].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])

                    # store model metadata
                    cacheMetadata['asofdate'].append(self.dataframe.index[i])
                    cacheMetadata['aic'].append(res.aic)
                    cacheMetadata['bic'].append(res.bic)
                    cacheMetadata['hqic'].append(res.hqic)
                    cacheMetadata['mae'].append(np.mean(np.abs(res.resid)))
                    cacheMetadata['mse'].append(np.mean(np.square(res.resid)))
                    cacheMetadata['rsquare'].append(np.mean(np.square(res.resid))/np.var(np.exp(diff[i:self.trainDFLength+i])))

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

                except:
                    cacheForecasts['pointForecast'].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])

                i+=self.forecastHorizon

            # Fixing data formats
            cacheForecasts['pointForecast'] = np.concatenate(cacheForecasts['pointForecast'])
            cacheForecasts['forecastday']   = np.array(self.forecastList*(int(len(cacheForecasts['pointForecast'])/int(self.forecastHorizon))))
            cacheForecasts[self.column]     = np.array(self.dataframe[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])])
            cacheForecasts['asofdate']      = np.array(self.dataframe[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])].index)
            cacheForecasts['product_name']  = np.array([self.product]*int(len(cacheForecasts['pointForecast'])))

            cacheForecasts['MA_20']          = self.dataframe[self.column].rolling(20).mean()[1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_50']          = self.dataframe[self.column].rolling(50).mean()[1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_200']         = self.dataframe[self.column].rolling(200).mean()[1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]

            cacheForecasts['MA_diff_20']          = self.dataframe[self.column].rolling(20).mean().diff(1)[1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_diff_50']          = self.dataframe[self.column].rolling(50).mean().diff(1)[1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_diff_200']         = self.dataframe[self.column].rolling(200).mean().diff(1)[1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]

            print(len(cacheForecasts['pointForecast']))
            print(len(cacheForecasts['forecastday']))
            print(len(cacheForecasts[self.column]))
            print(len(cacheForecasts['asofdate']))
            print(len(cacheForecasts['product_name']))

            pd.DataFrame(cacheForecasts).to_csv(f'forecasts_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}.csv')
            pd.DataFrame(cacheMetadata).to_csv(f'metadata_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}.csv')

        if not self.diff:
            diff   = self.dataframe[self.column].diff(1).dropna(axis=0)
    
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
                    cacheForecasts['pointForecast'].append(np.exp(fcast[0])*self.dataframe[self.column][self.trainDFLength+i])
                    cacheForecasts[self.column].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])

                    # store model metadata
                    cacheMetadata['asofdate'].append(self.dataframe.index[i])
                    cacheMetadata['aic'].append(res.aic)
                    cacheMetadata['bic'].append(res.bic)
                    cacheMetadata['hqic'].append(res.hqic)
                    cacheMetadata['mae'].append(np.mean(np.abs(res.resid)))
                    cacheMetadata['mse'].append(np.mean(np.square(res.resid)))
                    cacheMetadata['rsquare'].append(np.mean(np.square(res.resid))/np.var(np.exp(diff[i:self.trainDFLength+i])))

                    # event_data
                    cacheMetadata['forecastHorizon'].append(self.forecastHorizon)
                    cacheMetadata['diff'].append(self.diff)
                    cacheMetadata['trainDFLength'].append(self.trainDFLength)
                    cacheMetadata['num_models'].append(self.num_models)
                    cacheMetadata['order_p'].append(self.order[0])
                    cacheMetadata['order_q'].append(self.order[1])

                    for param in range(1,1+self.order[0]):
                        cacheMetadata[f'ar.L{param}.{self.column}'].append(res.bse[f'ar.L{param}.{self.column}'])

                    for param in range(1,1+self.order[1]):
                        cacheMetadata[f'ma.L{param}.{self.column}'].append(res.bse[f'ma.L{param}.{self.column}'])

                    cacheMetadata['time'].append(datetime.now() - start)

                except:
                    cacheForecasts['pointForecast'].append(self.dataframe[self.column][self.trainDFLength+i+1:self.trainDFLength+self.forecastHorizon+i+1])

                i+=self.forecastHorizon

            # Fixing data formats
            cacheForecasts['pointForecast'] = np.concatenate(cacheForecasts['pointForecast'])
            cacheForecasts['forecastday']   = np.array(self.forecastList*(int(len(cacheForecasts['pointForecast'])/int(self.forecastHorizon))))
            cacheForecasts[self.column]     = np.array(self.dataframe[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])])
            cacheForecasts['asofdate']      = np.array(self.dataframe[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])].index)

            cacheForecasts['MA_20']          = self.dataframe[self.column].rolling(20).mean()[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_50']          = self.dataframe[self.column].rolling(50).mean()[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_200']         = self.dataframe[self.column].rolling(200).mean()[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]

            cacheForecasts['MA_diff_20']          = self.dataframe[self.column].rolling(20).mean().diff(1)[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_diff_50']          = self.dataframe[self.column].rolling(50).mean().diff(1)[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]
            cacheForecasts['MA_diff_200']         = self.dataframe[self.column].rolling(200).mean().diff(1)[self.column][1+self.trainDFLength:1+self.trainDFLength+len(cacheForecasts['pointForecast'])]

            if self.b_adjust:
                pd.DataFrame(cacheForecasts).to_csv(f'forecasts_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}_{self.b_adjust}.csv')
                pd.DataFrame(cacheMetadata).to_csv(f'metadata_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}_{self.b_adjust}.csv')
            else:
                pd.DataFrame(cacheForecasts).to_csv(f'forecasts_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}.csv')
                pd.DataFrame(cacheMetadata).to_csv(f'metadata_{self.product}_{self.order}_{self.diff}_{self.forecastHorizon}.csv')           

arma_model_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\approved_strategies\corn_arma_ma\data\CBOT_DL_ZC1!, 1D.csv',
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(3,3),
    'num_models':13000,
    'diff':True,
    'product':'corn',
    'column':'close',
    'b_adjust':True,

}

model = train_arma(arma_model_event)

model.arma_model()