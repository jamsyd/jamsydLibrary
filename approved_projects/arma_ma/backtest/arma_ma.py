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

product = "VIX"

arma_model_event = {

    'dataframe':r"C:\Users\James Stanley\Downloads\CME_DL_CSC1!, 1D.csv",
    'forecastHorizon':5,
    'trainDFLength':252,
    'order':(1,0,0),
    'num_models':810,
    'diff':True,
    'product':product,
    'column':'close',

}

# Train the model
model = train_arma(arma_model_event)
model.arma_model()


pnl_event = {

    'product_name':product,
    'forecastHorizon':5,
    'dataframe':f"""{product}_forecasts_{str(arma_model_event['order'])}.csv""",
    'threshold':0.00,
    'reinvest':True,
    'strategy':'arma_ma',
}

# Backtest strategy
spread_threshold(pnl_event)

# analysis_event = {

#     'forecast_df':f"""forecasts_{product}_(2, 1)_True_5.csv""",
#     'pnl_df':f"""forecasts_{product}_(2, 1)_True_5.csv""",
#     'metadata_df':f"""metadata_{product}_(2, 1)_True_5.csv""",

# }

# # Build Analysis frames
# analysis_frames(analysis_event)