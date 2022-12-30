import datetime
import pandas as pd
import matplotlib.pyplot as plt

class seasonality:

    def __init__(self,event):
        self.dataframe = pd.read_csv(event['dataframe'],parse_dates=True)
        self.format    = event['format']
        self.plot      = event['plot']

    def seasonal_analysis(self):

        cacheYears    = [] 
        cacheDateTime = []

        for date in self.dataframe['time']:
            dateObject = date.split("T")[0]

            dateObject = datetime.datetime.strptime(dateObject, format).date()

            cacheDateTime.append(dateObject)

            if str(dateObject.year) not in cacheYears:
                cacheYears.append(str(dateObject.year))

        self.dataframe['time'] = cacheDateTime

        dataDict = {}

        for year in cacheYears:
            dataDict[year] = []

        for year in cacheYears:
            i = 0
            while i < len(self.dataframe):
                if year in str(self.dataframe['time'][i]):
                    dataDict[year].append(self.dataframe['close'][i])
                i+=1

        seasonality = pd.DataFrame.from_dict(dataDict,orient = 'index')
        seasonality = seasonality.T

        for year in seasonality.columns:
            seasonality[year] = seasonality[year].div(seasonality[year][0])

        seasonality.fillna(method='ffill')

        if self.plot:
            # Plot of mean values
            plt.figure(figsize=(45,15))

            plt.title('Mean',fontsize=50)

            plt.plot(seasonality.mean(axis=1)[0:250],color='blue',linewidth='2')
            plt.plot(seasonality.mean(axis=1)[0:250]+seasonality.var(axis=1)[0:250],color='red',linewidth='5')
            plt.plot(seasonality.mean(axis=1)[0:250]-seasonality.var(axis=1)[0:250],color='red',linewidth='5')

            plt.legend(['Mean','+1std','-1std'])

            plt.show()

        return seasonality
