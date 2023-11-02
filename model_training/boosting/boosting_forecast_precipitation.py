import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import xgboost as xgb

def reaing_data(station_path,cut_proportion = 0.85, n_lags=12):
    df = pd.read_csv(station_path, index_col = 'Data')
    df.index = pd.to_datetime(df.index)


    #Creating new features used in xgboost
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Week'] = df.index.isocalendar().week

    #Changing NaN for the mean of the column
    my_imputer = SimpleImputer(strategy = 'mean')
    imputed_df = pd.DataFrame(my_imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index

    #Creating the lags
    for i in range(1, n_lags + 1):
        imputed_df[f'lag_{i}'] = df['Precipitacao'].shift(i)
 
    #Defining the train/test split
    size = imputed_df.shape
    cut = round(cut_proportion*size[0])
    imputed_train = imputed_df.iloc[:cut,:]
    imputed_test = imputed_df.iloc[cut:,:]

    return imputed_train, imputed_test

def bosting(train_data, test_data):

    #Getting the number of days since the first day
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    start_date = train_data['Data'][0]
    train_data['Data'] = train_data['Data'].apply(lambda x: (x - start_date).days)
    test_data['Data'] = test_data['Data'].apply(lambda x: (x - start_date).days)

    #These are the columns that we will use in the model
    valid_columns = ['Data','lag_1','lag_2','lag_3','lag_4','lag_5',
                     'lag_6','lag_7', 'lag_8','lag_9','lag_10','lag_11','lag_12']
    
    X_train = train_data[valid_columns]
    y_train = train_data['Precipitacao']
    X_test = test_data[valid_columns]
    y_test = test_data['Precipitacao']

    #fiting the model
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.008, random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)

    #Creating a dataframe with the results
    result_df = pd.DataFrame({'Data': test_data['Data'],'Predicao': y_pred, 'Real': y_test})
    
    #Getting the date back
    result_df['Data'] = result_df['Data'].apply(lambda x: start_date + pd.DateOffset(days=int(x)))

    return result_df
        
def main():
    train_data, test_data = reaing_data(station_path = r'files\daily_data\A601.csv')
    result_df = bosting(train_data, test_data)

if __name__ == "__main__":
    main()