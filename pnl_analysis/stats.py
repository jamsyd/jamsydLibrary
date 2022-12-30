import numpy as np
import pandas as pd

class performance_analysis:

    def __init__():
        self.pnl = event['pnl']

    def hitrate(self):

        up = 0
        down = 0
        for p in pnl:

            if p < 0:
                down+=1
            if p > 0:
                up+=1

        return up / (up + down)
        
    def relativepnl(self):

        up = 0
        down = 0
        for p in self.pnl:

            if p < 0:
                down+=p
            if p > 0:
                up+=p
        return up / (down)

    def roc(pnl):
        return pd.DataFrame(pnl).pct_change().dropna()

    def get_realized_vol(dataset, time,col_name='pnl'):
        dataset['returns'] = np.log(dataset[col_name].cumsum()/dataset[col_name].cumsum().shift(1))
        dataset.fillna(0, inplace = True)
        volatility = dataset.returns.rolling(window=time).std(ddof=0)*np.sqrt(252)
        return volatility    