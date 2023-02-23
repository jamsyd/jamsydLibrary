import yaml
from yaml.loader import BaseLoader
from ProjectLibrary.seasonal_backtest import seasonal_test

with open(r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\approved_projects\seasonal_arma\data\input.yaml', 'r') as stream:
    input_data = yaml.load(stream,yaml.BaseLoader)


prdlist = ["CORN","SOYB","CANE"]

import multiprocessing
from multiprocessing import Pool

# create multiprocessing pool
# p = Pool(multiprocessing.cpu_count())
    
for product in prdlist:

    # reformat data
    input_data[product]['dataframe']       = str(input_data[product]['dataframe'])
    input_data[product]['forecastHorizon'] = int(input_data[product]['forecastHorizon'])
    input_data[product]['trainDFLength']   = int(input_data[product]['trainDFLength'])
    input_data[product]['num_models']      = int(input_data[product]['num_models'])       
    input_data[product]['diff']            = True
    input_data[product]['product']         = str(input_data[product]['product'])
    input_data[product]['Column']          = str(input_data[product]['Column'])
    input_data[product]['b_adjust']        = str(input_data[product]['b_adjust'])
    input_data[product]['order']           = tuple(map(int,input_data[product]['order'].split(",")))
    input_data[product]['b_adjust']        = str(input_data[product]['b_adjust'])
    
    # # cacheing dataframe
    # p.map(seasonal_test, )

    seasonal_test(input_data[product])