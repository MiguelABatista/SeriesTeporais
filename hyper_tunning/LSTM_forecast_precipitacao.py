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
    # Identificando NaN
    print(df1.describe())
    print(df1.isna().sum())

if __name__ == '__main__':
    file_path = r'meteorologia\concatenados_rio\A601.csv'
    reading_and_cleaning(file_path)

exit()
my_imputer = SimpleImputer(strategy = 'mean')
imputed_df1 = pd.DataFrame(my_imputer.fit_transform(df1))
imputed_df1.columns = df1.columns
imputed_df1.index = df1.index


# Definindo o corte e treino/teste
size = imputed_df1.shape
corte = round(0.85*size[0])
imputed_train = imputed_df1.iloc[:corte,0]
imputed_test = imputed_df1.iloc[corte:,0]


sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(np.array(imputed_train).reshape(-1,1))
test_set_scaled = sc.fit_transform(np.array(imputed_test).reshape(-1,1))
X_train = []
y_train = []

ws = 48

for i in range(ws, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-ws:i])
    y_train.append(training_set_scaled[i])
    
X_train, y_train = np.array(X_train), np.array(y_train)


Model_p = Sequential()

Model_p.add(LSTM(units = 64, return_sequences=True, input_shape = (X_train.shape[1],1)))
#Model_p.add(Dropout(0.2))

Model_p.add(LSTM(units = 32, return_sequences=True))
#Model_p.add(Dropout(0.1))

Model_p.add(LSTM(units = 32, return_sequences=True))

Model_p.add(LSTM(units = 32, return_sequences=True))
#Model_p.add(Dropout(0.2))

Model_p.add(LSTM(units = 16, return_sequences=True))
#Model_p.add(Dropout(0.2))

#Model_p.add(LSTM(units = 16, return_sequences=True))

Model_p.add(LSTM(units = 16))
#Model_p.add(Dropout(0.2))

Model_p.add(Dense(units=1))

Model_p.compile(optimizer = 'adam', loss = 'mean_squared_error')

Model_p.fit(X_train, y_train, epochs = 200, batch_size = 16)

# Salvando o modelo
#Model_p.save('LSTM-Univariate')

#from keras.models import load_model
#Model_p = load_model('LSTM-Univariate')

plt.plot(range(len(Model_p.history.history['loss'])),Model_p.history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# Forecast do modelo


prediction_test = []

Batch_one = training_set_scaled[-ws:]
Batch_new = Batch_one.reshape((1,ws,1))

for _ in range(len(test_set_scaled)):
    
    first_pred = Model_p.predict(Batch_new)[0]
    
    prediction_test.append(first_pred)
    
    Batch_new = np.append(Batch_new[:,1:,:],[[first_pred]], axis=1)
    
prediction_test = np.array(prediction_test)
predictions = sc.inverse_transform(prediction_test)

plt.plot(imputed_test.index, imputed_test, color = 'red', label = 'series')
plt.plot(imputed_test.index, predictions, color = 'blue', label = 'predicted values')
plt.title('LSTM - forecast')
plt.xlabel('Time')
plt.ylabel('Solar Irradiance')
plt.legend()
plt.show()

rmse_lstm = np.sqrt(mean_squared_error(predictions, imputed_test))
