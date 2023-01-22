
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def plot_acf_pacf(data):
    
    plot_acf(data, alpha=1, lags=20).savefig("acf.jph")
    plot_pacf(data, alpha=1, lags=20).savefig("pacf.jpg")
