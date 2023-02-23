def seasonal_test(seasonal_event):

    import pandas as pd
    from statsmodels.tsa.seasonal import seasonal_decompose
    from ProjectLibrary.multiplicative_residuals import trainResidualSeasonality

    # reading in dataframe
    seasonal_event['dataframe'] = pd.read_csv(seasonal_event['dataframe'],parse_dates=True,index_col='time') 
    seasonal_result             = seasonal_decompose(seasonal_event['dataframe']['close'], model="multiplicative",period=252)

    # defining seasonaltiy breakdown
    trend    = seasonal_result.trend
    seasonal = seasonal_result.seasonal
    residual = seasonal_result.resid

    # Seasonal event
    seasonal_event['column']    = 'resid'
    seasonal_event['dataframe'] = residual.reset_index().set_index(['time'])

    # Here we train the model
    model       = trainResidualSeasonality(seasonal_event)
    arma_model  = model.arma_model()
    forecast_df = arma_model[0]

    merge_df = pd.concat([forecast_df,trend.reset_index().set_index(['time'])['trend'],seasonal.reset_index().set_index(['time'])['seasonal']],axis=1)
    merge_df = merge_df.dropna(axis=0)

    merge_df['pointForecast'] = merge_df['residualForecast']*merge_df['trend']*merge_df['seasonal']
    merge_df                  = merge_df[['pointForecast','residualForecast','trend','seasonal','resid','product_name']]

    merge_df.to_csv(f"""{seasonal_event['product']}.csv""")

    # return (merge_df,arma_model[0],arma_model[1])