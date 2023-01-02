import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class seasonality:

    def __init__(self,event):

        # Formatting dataframe
        self.dataframe          = pd.read_csv(event['dataframe'],parse_dates=True)
        self.column_name        = event['column_name']
        self.format             = event['format']
        self.date_col           = event['date_col']


        # Plots
        self.plot_annual_performance = event['plot_annual_performance']
        self.plot_seasonality        = event['plot_seasonality']
        self.seasonality             = None
        self.annual_performance      = event['annual_performance']


    def annual_decomposition(self):

        cacheYears    = [] 
        cacheDateTime = []

        for date in self.dataframe[self.date_col]:

            dateObject = datetime.datetime.strptime(date.split(" ")[0], self.format).date()

            cacheDateTime.append(dateObject)

            if str(dateObject.year) not in cacheYears:
                cacheYears.append(str(dateObject.year))

        self.dataframe[self.date_col] = cacheDateTime

        dataDict = {}

        for year in cacheYears:
            dataDict[year] = []

        for year in cacheYears:
            i = 0
            while i < len(self.dataframe):
                if year in str(self.dataframe[self.date_col][i]):
                    dataDict[year].append(self.dataframe[self.column_name][i])
                i+=1

        self.seasonality = pd.DataFrame.from_dict(dataDict,orient = 'index')
        self.seasonality = self.seasonality.T

        for year in self.seasonality.columns:
            self.seasonality[year] = self.seasonality[year].div(self.seasonality[year][0])

        self.seasonality = self.seasonality.fillna(method='ffill')

        self.seasonality = self.seasonality[np.isfinite(self.seasonality)]

        self.seasonality.to_csv(r'seasonality.csv')

            
        if self.plot_seasonality:
 
            plt.figure(figsize=(45,15))

            plt.title('Mean',fontsize=50)

            plt.plot(self.seasonality.mean(axis=1),color='blue',linewidth='2')
            plt.plot(self.seasonality.mean(axis=1)+self.seasonality.var(axis=1),color='red',linewidth='5')
            plt.plot(self.seasonality.mean(axis=1)-self.seasonality.var(axis=1),color='red',linewidth='5')

            plt.legend(['Mean','+1std','-1std'])

            plt.show()

        if self.plot_annual_performance:
            plt.figure(figsize=(45,55))

            legend = []

            for column in self.seasonality.columns:
                plt.plot(self.seasonality[column])
                legend.append(str(column))

            plt.title('Annual Performance',fontsize=45)
            plt.legend(legend,fontsize=15)
            plt.show()


        if self.annual_performance:

            cacheAnnaualPerformance = {

                'year':[],
                'performance':[],

            }

            for column in self.seasonality.columns:

                cacheAnnaualPerformance['year'].append(column)
                cacheAnnaualPerformance['performance'].append(self.seasonality[column][len(self.seasonality[column])-1])
            
            self.annual_performance = pd.DataFrame(cacheAnnaualPerformance)

            print(self.annual_performance)

        return self


# seasonality_event = {

#     'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\Output\arma\corn\cumulative_pnl.csv',
#     'format':'%Y-%m-%d',
#     'plot_annual_performance':True,
#     'plot_seasonality':True,
#     'annual_performance':True,
#     'column_name':'pnl',
#     'date_col':'asofdate',

# }

