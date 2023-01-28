"""
    Description: Program designed to train ARMA model and estimate the spread_threshold strategy. 
"""

import numpy as np
import pandas as pd

from ProjectLibrary.arma import train_arma
from ProjectLibrary.arma_threshold_ma import spread_threshold
from ProjectLibrary.produce_analysis import analysis_frames

arma_model_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\data\daily\stocks\BATS_CORN, 1D.csv',
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(2,2),
    'num_models':2500,
    'diff':True,
    'product':'BATS_CORN',
    'column':'close',
    'b_adjust':True,

}


pnl_event = {

    'forecastHorizon':5,
    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\forecasts_BATS_CORN_(2, 2)_True_5.csv',
    'threshold':0.01,
    'reinvest':True,
    'strategy':'arma_threshold_ma',
}

analysis_event = {

    'forecast_df':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\forecasts_BATS_CORN_(2, 2)_True_5.csv',
    'pnl_df':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\pnl_armaspreadthreshold_BATS_CORN_(2, 2)_True_5.csv',
    'metadata_df':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\metadata_BATS_CORN_(2, 2)_True_5.csv',

}


# # Train the model
# model = train_arma(arma_model_event)
# model.arma_model()

# Backtest strategy
spread_threshold(pnl_event)

# Build Analysis frames
analysis_frames(analysis_event)