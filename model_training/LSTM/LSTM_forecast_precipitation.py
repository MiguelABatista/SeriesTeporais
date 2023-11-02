# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

def reading_and_cleaning(file_path):
    df = pd.read_csv(file_path, index_col = 'Data')
    df.index = pd.to_datetime(df.index)

    # Passando para dados mensais
    df1 = df.resample('M').mean()

    # Removendo os NaN
    my_imputer = SimpleImputer(strategy = 'mean')
    imputed_df1 = pd.DataFrame(my_imputer.fit_transform(df1))
    imputed_df1.columns = df1.columns
    imputed_df1.index = df1.index

    return imputed_df1

def train_test_split(df, windows_size):
    size = df.shape
    corte = round(0.85*size[0])
    imputed_train = df.iloc[:corte,0]
    imputed_test = df.iloc[corte:,0]


    scaler = MinMaxScaler(feature_range=(0,1))

    training_set_scaled = scaler.fit_transform(np.array(imputed_train).reshape(-1,1))
    test_set_scaled = scaler.fit_transform(np.array(imputed_test).reshape(-1,1))
    X_train = []
    y_train = []

    for i in range(windows_size, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-windows_size:i])
        y_train.append(training_set_scaled[i])
        
    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train, imputed_test, test_set_scaled,training_set_scaled, scaler

def define_model(X_train, neurons_list):
    Model_p = Sequential()

    lenth = len(neurons_list)

    for index, pair in enumerate(neurons_list):
        neurons_amount, dropout_rate = pair

        if index == 1:
            Model_p.add(LSTM(units = neurons_amount, return_sequences=True, input_shape = (X_train.shape[1],1)))
            continue

        if index == lenth-1:
            Model_p.add(LSTM(units = neurons_amount))
            break

        Model_p.add(LSTM(units = neurons_amount, return_sequences=True))
        
        if dropout_rate != 0:
            Model_p.add(Dropout(dropout_rate))
   
    Model_p.add(Dense(units=1))
    Model_p.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return Model_p

def model_forecast(Model_p, training_set_scaled, test_set_scaled, scaler, windows_size):
    prediction_test = []

    Batch_one = training_set_scaled[-windows_size:]
    Batch_new = Batch_one.reshape((1,windows_size,1))

    for _ in range(len(test_set_scaled)):
        
        first_pred = Model_p.predict(Batch_new)[0]
        
        prediction_test.append(first_pred)
        
        Batch_new = np.append(Batch_new[:,1:,:],[[first_pred]], axis=1)
        
    prediction_test = np.array(prediction_test)
    predictions = scaler.inverse_transform(prediction_test)

    return predictions

def main():
    file_path = r'meteorologia\diarios_rio\A601.csv'
    windows_size = 48

    df = reading_and_cleaning(file_path)

    X_train, y_train, imputed_test, test_set_scaled, training_set_scaled, scaler = train_test_split(df, windows_size)

    neurons_list = [(64,0),(32,0),(32,0),(32,0),(16,0),(16,0),(1,0)]
    Model_p = define_model(X_train, neurons_list)

    Model_p.fit(X_train, y_train, epochs = 10, batch_size = 16)

    plt.plot(range(len(Model_p.history.history['loss'])),Model_p.history.history['loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()  

    predictions = model_forecast(Model_p, training_set_scaled, test_set_scaled, scaler, windows_size)

    plt.plot(imputed_test.index, imputed_test, color = 'red', label = 'series')
    plt.plot(imputed_test.index, predictions, color = 'blue', label = 'predicted values')
    plt.title('LSTM - forecast')
    plt.xlabel('Time')
    plt.ylabel('Solar Irradiance')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
