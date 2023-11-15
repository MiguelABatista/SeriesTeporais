import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb


df = pd.read_csv('A601.csv', index_col = 'Data')
df.index = pd.to_datetime(df.indexpytho)

# Passando para dados mensais
df1 = df.resample('M').mean()
# Identificando NaN
print(df1.describe())
print(df1.isna().sum())


#Criando novas features usadas no xgboost
df1['Month'] = df1.index.month
df1['Quarter'] = df1.index.quarter
df1['Week'] = df1.index.week

my_imputer = SimpleImputer(strategy = 'mean')
imputed_df1 = pd.DataFrame(my_imputer.fit_transform(df1))
imputed_df1.columns = df1.columns
imputed_df1.index = df1.index


def create_supervised_data(data, n_lags=1):
    df = pd.DataFrame(data)
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Precipitacao'].shift(i)
    df.dropna(inplace=True)
    return df

n_lags=12

imputed_df1 = create_supervised_data(imputed_df1, n_lags)


# Definindo o corte e treino/teste
size = imputed_df1.shape
corte = round(0.85*size[0])
imputed_train = imputed_df1.iloc[:corte,:]
imputed_test = imputed_df1.iloc[corte:,:]

# Preenchendo os NaN com o SimpleImputer
'''
my_imputer = SimpleImputer(strategy = 'mean')
imputed_train = pd.DataFrame(my_imputer.fit_transform(train))
imputed_test = pd.DataFrame(my_imputer.transform(test))

imputed_train.columns = train.columns
imputed_test.columns = test.columns

imputed_train.index = train.index
imputed_test.index = test.index
'''
#print(imputed_train.head())

imputed_train['Precipitacao'].plot()

print(adfuller(imputed_train['Precipitacao']))
# Criando a precipitação diferenciada 
#imputed_train['Precipitacao_diff'] = imputed_train.Precipitacao.diff()
#imputed_train = imputed_train.dropna()

# criando um benchmark: a média
imputed_test['benchmark'] = imputed_train['Precipitacao'].mean()


# plotando a série de treino e de teste ambas imputadas
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,5),sharex=True)
ax.plot(imputed_train['Precipitacao'], label = 'train')
ax.plot(imputed_test['Precipitacao'], label= 'test')
ax.plot(imputed_test['benchmark'], linestyle='--', label = 'benchmark - média')
ax.axvspan(imputed_train.index[-1],imputed_test.index[-1],color='#808080',alpha=0.4)
plt.ylim(0, 0.55)
plt.xlim(imputed_train.index[0],imputed_test.index[-1])
plt.grid(True)
plt.legend()
plt.title('Treino x Teste')

rmse_bench = np.sqrt(mean_squared_error(imputed_test['Precipitacao'], imputed_test['benchmark']))


#imputed_train['Precipitacao_diff'].plot()
#print(adfuller(imputed_train['Precipitacao_diff']))

# Holt Winters

# Valores precisam ser estritamente positivos, logo substituimos zero por pequenos
imputed_train['Precipitacao_hw'] = imputed_train['Precipitacao'].replace(0,1e-12)
hw = ExponentialSmoothing(imputed_train['Precipitacao_hw'],
                          trend='add',seasonal='add',seasonal_periods=12, initialization_method = 'legacy-heuristic')
res_h = hw.fit()
hw_fitted = res_h.fittedvalues
hw_forecast = res_h.forecast(len(imputed_test['Precipitacao']))
rmse_hw_test = np.sqrt(mean_squared_error(imputed_test['Precipitacao'], hw_forecast))
rmse_hw_train = np.sqrt(mean_squared_error(imputed_train['Precipitacao'], hw_fitted))

fig, ax = plt.subplots(figsize = (15,5))
ax.plot(imputed_train.index, imputed_train['Precipitacao'], label='train')
ax.plot(imputed_train.index, hw_fitted,label = 'fitted')
ax.plot(imputed_test.index, imputed_test['Precipitacao'],label='test')
ax.plot(imputed_test.index, hw_forecast,label='forecast',linestyle = '--')
plt.ylim(0, 0.55)
plt.xlim(imputed_train.index[0],imputed_test.index[-1])
plt.grid(True)
ax.legend()
plt.title('Holt Winters: ' + 'RMSE Train set = ' + str(round(rmse_hw_train,3) ) + ' | '+ 'RMSE Test set = ' + str(round(rmse_hw_test,3)))

"""
# plotando a autocorrelação
fig, ax = plt.subplots(figsize = (15,5))
plot_acf(np.array(imputed_train['Precipitacao']), ax=ax,lags=40);

fig, ax = plt.subplots(figsize = (15,5))
plot_pacf(np.array(imputed_train['Precipitacao']), ax=ax,lags=40);

"""

# rodando o grid search com o auto arima
model_1 = pm.auto_arima(imputed_train['Precipitacao'], trace=True, supress_warnings = True, seasonal = True,
                      stepwise = False, max_p = 16, max_d=2, max_q = 9, max_order=30, m=12)

#fig2, ax2 = plt.subplots(figsize=(10,6))

# diagnostico dos resíduos
model_1.plot_diagnostics();

# Plots das pevisões
test_pred, confint = model_1.predict(len(imputed_test), return_conf_int = True)

fig, ax = plt.subplots(figsize = (10,5))
ax.plot(imputed_test.index, imputed_test['Precipitacao'], label='test', marker='o')
ax.plot(imputed_test.index, test_pred, label='forecast', marker = 'x')
ax.fill_between(imputed_test.index, confint[:,0], confint[:,1], color = 'red', alpha = 0.3)
plt.grid(True)
ax.legend();

train_pred = model_1.predict_in_sample(start = 0, end = -1)

# Calculando o RMSE
rmse_train = np.sqrt(mean_squared_error(imputed_train.iloc[:]['Precipitacao'],train_pred))
rmse_test = np.sqrt(mean_squared_error(imputed_test['Precipitacao'], test_pred))
rmse_bench = np.sqrt(mean_squared_error(imputed_test['Precipitacao'], imputed_test['benchmark']))

fig, ax = plt.subplots(figsize = (15,5))
ax.plot(imputed_df1.index, imputed_df1['Precipitacao'], label='data')
ax.plot(imputed_train.index[:], train_pred, label='fitted')
ax.plot(imputed_test.index, test_pred, label='forecast')
ax.plot(imputed_test['benchmark'], linestyle='--',label='mean')
ax.fill_between(imputed_test.index, confint[:,0], confint[:,1], color = 'red', alpha = 0.3)
plt.ylim(0, 0.55)
plt.xlim(imputed_train.index[0],imputed_test.index[-1])
plt.grid(True)
plt.title('RMSE Train set = ' + str(round(rmse_train,3) ) +' | '+ 'RMSE Test set = ' + str(round(rmse_test,3) ))

ax.legend();

# XGBoost
X_train = imputed_train[['Month','Quarter','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7',
                        'lag_8','lag_9','lag_10','lag_11','lag_12']]
y_train = imputed_train['Precipitacao']
X_test = imputed_test[['Month','Quarter','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7',
                        'lag_8','lag_9','lag_10','lag_11','lag_12']]
y_test = imputed_test['Precipitacao']

model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.008, random_state=42)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(imputed_test.index, y_test, label='test', marker='o')
plt.plot(imputed_test.index, y_pred, label='forecast', marker='x')
#plt.xlabel('Data')
#plt.ylabel('Valor')
plt.title('XGBoost')
plt.legend()
plt.grid(True)
plt.show()

rmse_xgb = np.sqrt(mean_squared_error(imputed_test['Precipitacao'], y_pred))


fig, ax = plt.subplots(figsize = (15,5))
ax.plot(imputed_test.index, y_test, label='test', marker='o')
ax.plot(imputed_test.index, y_pred, label='forecast-XGBoost', marker='x')
ax.plot(imputed_test.index, test_pred, label='forecast-SARIMA', marker = 'v')
plt.title('Comparação SARIMA x XGBoost')
plt.legend()
plt.grid(True)
plt.show()

