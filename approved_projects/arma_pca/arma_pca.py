import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.multivariate.pca import PCA

# Initial event of the model
pca_event = {

    'nComp':4,
    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\data\daily\commodities\soft_commodities\grains\pca_grains\CBOT_DL_GRAIN, 1D.csv',
    'constant_correlation':True,
    'column':'ZC1!, CBOT: Close',
}


def pca_model(pca_event):

    """
        Description: 
    """

    # reading the dataframe
    pca_df = pd.read_csv(pca_event['dataframe'],index_col='time', parse_dates=True)

    # making sure the time series is stationary
    pca_logDiff_df = np.log(pca_df).diff(1).dropna(axis=0).fillna(0)

    # Perform principle component analysis
    pc = PCA(pca_logDiff_df,ncomp=pca_event['nComp'])

    # Generate forecasts
    forecast=0
    for i in range(0,pca_event['nComp']):
        forecast+= pc.eigenvecs['eigenvec_'+str(i)][i]*pc.factors['comp_'+str(i)] # I think this could be wrong here RESEARCH AND FIX

    # Plotting Corn Price
    plt.figure(figsize=(45,15))

    plt.title('Corn Price',fontsize=45)
    plt.plot(pca_df['ZC1!, CBOT: Close'])
    plt.plot(np.exp(forecast.cumsum())*pca_df['ZC1!, CBOT: Close'][0])

    plt.show()

    return forecast


pca_model(pca_event)

# class pca_model:

#     def __init__(self):
#         self.dataframe            = pd.read_csv(pca_event['dataframe'],index_col='time', parse_dates=True)
#         self.constant_correlation = pca_event['constant_correlation']
#         self.ncomp                = pca_event['nComp']
#         self.column               = pca_event['column']
    
#         if pca_event['forecast_dataframe'] is not None:
#             self.forecast = pca_event['forecast_dataframe']

#         self.log_diff             = np.log(self.dataframe).diff(1).dropna(axis=0).fillna(0)
#         self.trainDFLength        = 252
#         self.forecastHorizon      = 5

#     def constant_correlation(self):

#         """
#             Description: Train PCA Model

#         """

#         # Perform principle component analysis
#         pc = PCA(self.log_diff,ncomp=pca_event['nComp'])

#         # Generate forecasts
#         forecast=0
#         for i in range(0,pca_event['nComps']):
#             forecast+= pc.eigenvecs['eigenvec_'+str(i)][0]*pc.factors['comp_'+str(i)]

#         # Plotting Corn Price
#         plt.figure(figsize=(45,15))

#         plt.title('Corn Price',fontsize=45)
#         plt.plot(self.dataframe[self.column])
#         plt.plot(np.exp(forecast.cumsum())*self.dataframe[self.column][0])

#         plt.show()

#         return forecast


#     def pca_update(self):


#         horizon    = 30 # iterate through at horizon length
#         num_models = 300

#         forecast = []

#         i = 0
#         while i < len(num_models):

#             if i % horizon == 0:
#                 # train single pca model here
#                 pc = PCA(self.log_diff[self.trainDFLength],ncomp=pca_event['nComp'])

#             # Generate forecasts
#             forecast=0
#             for i in range(0,pca_event['nComps']):


#                 forecast+= pc.eigenvecs['eigenvec_'+str(i)][0]*pc.factors['comp_'+str(i)]

#             forecast.append(np.exp(forecast.cumsum())*self.dataframe[self.column][self.trainDFLength+i])

#             i+=self.forecastHorizon


#     def pca_arma(self):

#         """
#             Description: pca update instead we use the arma models for each of the principle components 
#                             but still train the prnciple components on the correlation with log diff. 
#         """


#         horizon    = 30 # iterate through at horizon length
#         num_models = 300

#         forecast = []

#         i = 0
#         while i < len(num_models):

#             if i % horizon == 0:
#                 # train single pca model here
#                 pc = PCA(self.log_diff[self.trainDFLength],ncomp=pca_event['nComp'])

#             # Generate forecasts
#             forecast=0
#             for i in range(0,pca_event['nComps']):

#                 forecast+= pc.eigenvecs['eigenvec_'+str(i)][0]*pc.factors['comp_'+str(i)]

#             forecast.append(np.exp(forecast.cumsum())*self.dataframe[self.column][self.trainDFLength+i])

#             i+=self.forecastHorizon



#     def pca_varma(self):

#         """
#             Description: Model definition:: ts_to_model ~ pca_model iteratively updated + ei + thetai. A vector autoregressive 
#                          model for the 
#         """


#         horizon    = 30 # iterate through at horizon length
#         num_models = 300

#         forecast = []

#         i = 0
#         while i < len(num_models):

#             if i % horizon == 0:
#                 # train single pca model here
#                 pc = PCA(self.log_diff[self.trainDFLength],ncomp=pca_event['nComp'])

#             # Generate forecasts
#             forecast=0
#             for i in range(0,pca_event['nComps']):


#                 forecast+= pc.eigenvecs['eigenvec_'+str(i)][0]*pc.factors['comp_'+str(i)]

#             forecast.append(np.exp(forecast.cumsum())*self.dataframe[self.column][self.trainDFLength+i])

#             i+=self.forecastHorizon

#     def pca_seasonal_arma():

#         """Description: Seasonal version of pca_arma --> use the residuals """


#     def pca_seasonal_varma():

#         """Description: Seasonal version of pca_arma --> use the residuals """


# pca = pca_model(pca_event)
# pca.constant_correlation()