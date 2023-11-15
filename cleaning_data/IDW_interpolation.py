# Montando o dataset

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import config

'''Montando nosso dataset a ser usado no kriging'''

def idw_interpolation(data, df_precipitacao, stations, station, coordinates, date, power=2, num_neighbors=6):
    
    distances = cdist(np.array([[coordinates.loc[station]['Latitude'], coordinates.loc[station]['Longitude']]]), 
                      coordinates[['Latitude', 'Longitude']].values)
    
    # Ordenar as distâncias e pegar os índices dos pontos mais próximos
    nearest_neighbors_indices = np.argsort(distances, axis=1)[:, :num_neighbors]
    
    # Adicionar uma pequena constante às distâncias para evitar divisões por zero
    epsilon = 1e-10
    distances += epsilon
    
    # Calcular as ponderações com base nas distâncias
    weights = 1 / np.power(distances, power)

    # Filtrar estações vizinhas com NaN para a data específica
    valid_indices = ~np.isnan(data.loc[date, df_precipitacao.columns[nearest_neighbors_indices[0]]])
    
    if valid_indices.any():
        # Calcular os valores interpolados usando apenas estações vizinhas válidas
        neighbors_data = data.loc[date, df_precipitacao.columns[nearest_neighbors_indices[0][valid_indices]]]
        valid_indices_1 = valid_indices[valid_indices]
        indices = [stations.index(x) for x in valid_indices_1.index]
        interpolated_value = np.sum(weights[0][indices] * neighbors_data) / np.sum(weights[0][indices])
        return interpolated_value
    else:
        # Lidar com o caso em que não há estações vizinhas válidas para a interpolação
        return np.nan

def read_data(files_path):
    stations = ['A601','A602','A603','A604','A606','A607','A608','A609','A610','A611',
                'A618','A619','A620','A621','A624','A625','A626','A627','A628','A629',
                'A630','A636','A637','A652','A659','A667']
    precipitation_columns = []

    #Here we will loop over every dataset getting the precipitation column
    for name in stations:
        df_path = os.path.join(files_path, name + '.csv')
        df = pd.read_csv(df_path, index_col = 'Data')
        df.index = pd.to_datetime(df.index)
        extract_column = df['Precipitacao']
        precipitation_columns.append(extract_column)

    precipitation_df = pd.concat(precipitation_columns, axis=1)
    precipitation_df.columns = stations
    #we will remove this station because it just have data from nov 2022, almost nothing
    precipitation_df.drop('A637', axis=1, inplace=True) 
    #We will get from this date because after it we have few station with data
    precipitation_df = precipitation_df.loc['2017-08-25 00:00:00':] 
    
    data = pd.read_csv('dados_geograficos_1.csv', index_col='Estacao')
    data.drop('A637', axis=0,inplace=True)

    stations.remove('A637')

    return precipitation_df, data, stations

def main():
    # Criando lista de datas onde faremos o loop:

    df_precipitacao, data, stations = read_data(config.daily_data_path)

    data_index = df_precipitacao[df_precipitacao.isna()].index

    for date in data_index:
        for station in df_precipitacao.columns:
            if pd.isna(df_precipitacao.loc[date, station]):
                interpolated_value = idw_interpolatin(data, df_precipitacao, stations, station, coordinates, date)
                df_precipitacao.loc[date, station] = interpolated_value
                
    df_precipitacao.to_csv('precipitacao_interpolada.csv', index=False)

if __name__ == '__main__':
    main()