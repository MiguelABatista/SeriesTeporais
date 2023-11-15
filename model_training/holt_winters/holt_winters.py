import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def reaing_data(station_path,cut_proportion = 0.85, n_lags=12):
    df = pd.read_csv(station_path, index_col = 'Data')
    df.index = pd.to_datetime(df.index)

    #Changing NaN for the mean of the column
    my_imputer = SimpleImputer(strategy = 'mean')
    imputed_df = pd.DataFrame(my_imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
 
    #Defining the train/test split
    size = imputed_df.shape
    cut = round(cut_proportion*size[0])
    imputed_train = imputed_df.iloc[:cut,:]
    imputed_test = imputed_df.iloc[cut:,:]

    return imputed_train, imputed_test

def holt_winters(imputed_train, imputed_test):
    #Replacing 0 for a very small number to avoid errors
    imputed_train['Precipitacao_hw'] = imputed_train['Precipitacao'].replace(0,1e-12)
    
    #Nao sei
    hw = ExponentialSmoothing(imputed_train['Precipitacao_hw'],trend='add',seasonal='add',seasonal_periods=12, initialization_method = 'legacy-heuristic')

    #Fitting the model
    res_h = hw.fit()
    hw_fitted = res_h.fittedvalues
    hw_forecast = res_h.forecast(len(imputed_test['Precipitacao']))
    
    return hw_fitted, hw_forecast

def main():
    pass

if __name__ == '__main__':
    main()
