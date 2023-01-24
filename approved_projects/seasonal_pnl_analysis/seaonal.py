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


        self.dataframe['pnl'] =  self.dataframe['pnl'].cumsum()

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

        ### Above here is where we need to make an adjustment 
            ## Need to be able to analyze positions by year, month based on different factors
        
        ##############################################################################################################################################
        # Generic plots
        ##############################################################################################################################################

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

        ##############################################################################################################################################
        # Performance Analytics based on Full PNL of each position
        ##############################################################################################################################################

        if self.annual_performance:

            cacheAnnaualPerformance = {

                'year':[],
                'eoy_performance':[],
                'worst_performance':[],
                'best_performance':[],
                'variance':[],
                'stdev':[],
                'sharpe':[],
                'perf_loss':[],
                'rfr':[],

            }

            def sharpe_ratio(pnl_dataframe,rfr):

                if 0 in pnl_dataframe:

                    return 0
                else:

                    log_return = np.log(pnl_dataframe-1).diff(1)

                    return (252*(np.mean(log_return) - rfr)/np.std(log_return))**0.5


            
            # Need to add dataframe for position based pnl
                # include hitrate, win_size, loss_size, hit_rate

            for column in self.seasonality.columns:

                cacheAnnaualPerformance['year'].append(column)
                cacheAnnaualPerformance['eoy_performance'].append(self.seasonality[column][len(self.seasonality[column])-1])
                cacheAnnaualPerformance['worst_performance'].append(np.min(self.seasonality[column]))
                cacheAnnaualPerformance['best_performance'].append(np.max(self.seasonality[column])) # return index of performance
                cacheAnnaualPerformance['variance'].append(np.var(self.seasonality[column])) # return index of performance
                cacheAnnaualPerformance['stdev'].append(np.std(self.seasonality[column])) # return index of performance
                cacheAnnaualPerformance['perf_loss'] = np.array(cacheAnnaualPerformance['best_performance']) - np.array(cacheAnnaualPerformance['eoy_performance'])
                cacheAnnaualPerformance['sharpe']    = sharpe_ratio(pnl_dataframe=self.seasonality[column].fillna(0),rfr=0.02)
                cacheAnnaualPerformance['rfr'].append(0.02)
                # Add hitrate back into this equation
            self.annual_performance = pd.DataFrame(cacheAnnaualPerformance) 

        print(self.annual_performance)

        ##############################################################################################################################################
        # Generic Trade Based Analytics
        ##############################################################################################################################################
        

        # Where forecastday = 5 take pnl at that point
            # Create dictionary where keys are a dictionary wit keys asofdate and pnl with corresponding pnl and date the pnl is calculated
            # Store trade dictionary as a json file

        
        cacheTradePerformance = {
            'year':[],
            'No.Trades':[],
            'relative_pnl':[],
            'hit_rate':[],
            'Minimum':[],
            'Maximum':[],
            'Mean':[],
            'Variance':[],
            'std_dev':[],
            'skew':[],
            'kurtosis':[],

        }

        trade_identification = {'asofdate':[],'pnl':[]}

        # Dynamiacally create a dictionary where keys are year and values are key

        # print("No. Trades: ",len(short_pnl['pnl']))
        # print("relative Pnl: ", relativepnl(short_pnl['pnl']))
        # print("Hit Rate:     ", hitrate(short_pnl['pnl']))
        # print("Minimum:      ",np.min(short_pnl['pnl']))
        # print("Maximum:      ",np.max(short_pnl['pnl']))
        # print("Mean:         ",np.mean(short_pnl['pnl']))
        # print("Variance:     ",np.var(short_pnl['pnl']))
        # print("Std. Dev:     ",np.sqrt(np.var(short_pnl['pnl'])))
        # print("Skew:         ",skew(short_pnl['pnl']))
        # print("Kurtosis:     ",kurtosis(short_pnl['pnl']))

        # Save model metadata 

        return self


seasonality_event = {

    'dataframe':r'C:\Users\James Stanley\Documents\GitHub\backtest_utilities\pnl_armaspreadthreshold_corn_(2, 2)_True_5.csv',
    'format':'%Y-%m-%d',
    'plot_annual_performance':True,
    'plot_seasonality':True,
    'annual_performance':True,
    'column_name':'pnl',
    'date_col':'asofdate',

}

seasonality(seasonality_event).annual_decomposition()