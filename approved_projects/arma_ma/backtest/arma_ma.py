"""
    Date:
    Author:
    Description: Program designed to train ARMA model and estimate the spread_threshold strategy. 
"""

import os

import numpy as np
import pandas as pd

from ProjectLibrary.arma import train_arma
from ProjectLibrary.arma_threshold_ma import spread_threshold
from ProjectLibrary.produce_analysis import analysis_frames

product = "CANE"

arma_model_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\data\daily\stocks\BATS_CANE, 1D.csv',
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(1,1),
    'num_models':2760,
    'diff':True,
    'product':product,
    'column':'close',
    'b_adjust':True,

}

# # # Train the model
model = train_arma(arma_model_event)
model.arma_model()


pnl_event = {

    'forecastHorizon':5,
    'dataframe':f"""forecasts_{product}_(1, 1)_True_5.csv""",
    'threshold':0.01,
    'reinvest':True,
    'strategy':'arma_ma',
}


# Backtest strategy
spread_threshold(pnl_event)

analysis_event = {

    'forecast_df':f"""forecasts_{product}_(1, 1)_True_5.csv""",
    'pnl_df':f"""forecasts_{product}_(1, 1)_True_5.csv""",
    'metadata_df':f"""metadata_{product}_(1, 1)_True_5.csv""",

}

# Build Analysis frames
analysis_frames(analysis_event)