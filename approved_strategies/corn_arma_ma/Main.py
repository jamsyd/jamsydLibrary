"""
    Description: Program designed to train ARMA model and estimate the spread_threshold strategy. 
"""

import numpy as np
import pandas as pd

from ProjectLibrary.arma import train_arma
from ProjectLibrary.arma_threshold_ma import spread_threshold
from ProjectLibrary.produce_analysis import analysis_frames

arma_model_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\approved_strategies\corn_arma_ma\data\CBOT_DL_ZC1!, 1D.csv',
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(3,3),
    'num_models':10,
    'diff':True,
    'product':'corn',
    'column':'close',
    'b_adjust':True,

}


pnl_event = {

    'forecastHorizon':5,
    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\forecasts_corn_(3, 3)_True_5.csv',
    'threshold':0.01,
    'reinvest':True,
    'strategy':'arma_threshold_ma',
}

# Train the model
model = train_arma(arma_model_event)
model.arma_model()

# Backtest strategy
spread_threshold(pnl_event)

# Build Analysis frames
analysis_frames()