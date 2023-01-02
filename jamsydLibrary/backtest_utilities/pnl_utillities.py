import numpy as np
import pandas as pd

def cumulative_pnl(pnl_vector,const):
    return 1+pnl_vector['pnl'].cumsum()/(const)
