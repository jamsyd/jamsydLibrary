import os
import sys

import numpy as np
import pandas as pd

from datetime import datetime
from statsmodels.tsa.arima_model import ARMA

single_arma_event = {

    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(2,2),
    'num_models':13000,

}

arma_df = pd.read_csv(r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\approved_strategies\corn_arma_ma\data\CBOT_DL_ZC1!, 1D.csv')

#################################################################################
# Identify dataframe - get dates we want to push to AWS
#################################################################################

cacheDatesArmaLambda = {

    'start_date':[],
    'end_date':[],
    'forecastHorizon':[],

}

# def retrieve_dates(cacheDatesArmaLambdaEvent):

i = 0
while i < single_arma_event['num_models']:

    cacheDatesArmaLambda['start_date'].append(arma_df['time'][i])
    cacheDatesArmaLambda['end_date'].append(arma_df['time'][single_arma_event['trainDFLength']+i])

    i+=single_arma_event['forecastHorizon']

arma_dates = pd.DataFrame(cacheDatesArmaLambda)

arma_dates.to_csv(r'arma_dates.csv')

    # return arma_dates


#############################################################################################################
# Train single arma model
#############################################################################################################

def lambda_event(arma_lambda_event):

    if arma_lambda_event['logDiff']:
        diff = np.log(arma_lambda_event['dataframe'])

    mod = ARMA(diff[arma_lambda_event['']], order=(2,2))
    # training the model
    res = mod.fit()
    
    # converting log forecast
    fcast = res.forecast(self.forecastHorizon)


diff = np.log(arma_lambda_event['dataframe']).diff(1)

i = 0
while i < len(arma_dates):

    print(arma_df['close'][(arma_df['time'] > arma_dates['start_date'][i]) and (arma_df['time'] > arma_dates['start_date'][i])])
    mod = ARMA(diff['close'], order=(2,2))



############################
# Push model to aws
############################

