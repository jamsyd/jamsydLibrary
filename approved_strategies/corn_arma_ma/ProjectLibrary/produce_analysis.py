import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis

def analysis_frames(analysis_event):

    # Reading in Data Fames
    forecast_df = pd.read_csv(analysis_event['forecast_df'],parse_dates=True,index_col='asofdate')
    pnl_df      = pd.read_csv(analysis_event['pnl_df'],parse_dates=True,index_col='asofdate')
    metadata    = pd.read_csv(analysis_event['metadata_df'],parse_dates=True,index_col='asofdate')

    def hitrate(pnl):

        up = 0
        down = 0
        for p in pnl:

            if p < 0:
                down+=1
            if p >= 0:
                up+=1

        return up / (up + down)


    def relativepnl(pnl):

        up = 0
        down = 0
        for p in pnl:

            if p < 0:
                down+=np.abs(p)
            if p >= 0:
                up+=np.abs(p)
                
        return up / (down)



    # Retrieve dataframe of positions
    merge_df = pnl_df.merge(forecast_df, left_on='asofdate', right_on='asofdate')
    positions = merge_df[(merge_df['forecastday'] == 5)][['pnl','pointForecast','forecastday','product_name']]

    # Positions Summary
    positions = positions[positions['pnl']!=0]
    performance_dict = {

        'No._Trades':[len(positions['pnl'])],
        'relative_Pnl':[relativepnl(positions['pnl'])],
        'Hit_Rate':[hitrate(positions['pnl'])],
        'Minimum':[np.min(positions['pnl'])],
        'Maximum':[np.max(positions['pnl'])],
        'Mean':[np.mean(positions['pnl'])],
        'Variance':[np.var(positions['pnl'])],
        'Std_Dev':[np.std(positions['pnl'])],
        'Skew':[np.float(skew(positions['pnl']))],
        'Kurtosis':[np.float(kurtosis(positions['pnl']))],
    }

    # Metadata Summary 
    metadata.mean(axis=0).to_csv(f"""metadatasummary_{metadata['product_name'][0]}_arma_({metadata['order_p'][0]},{metadata['order_q'][0]}).csv""")

    # List all positions
    positions.to_csv(f"""positions_{metadata['product_name'][0]}_arma_ma50_({metadata['order_p'][0]},{metadata['order_q'][0]}).csv""")

    # Position Summary CSV
    pd.DataFrame(performance_dict).to_csv(f"""tradeperformance_corn_arma_ma50_({metadata['order_p'][0]},{metadata['order_q'][0]}).csv""")
