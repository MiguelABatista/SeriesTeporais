import pandas as pd
import os

dfs_path = r'C:\Users\Miguel\Documents\GitHub\SeriesTeporais\meteorologia\diarios_rio'

spatial_df = pd.read_csv(r'C:\Users\Miguel\Documents\GitHub\SeriesTeporais\dados_geograficos.csv')
precipitation_df = pd.DataFrame()
def gera_df(coluna):
    for arquivo in os.listdir(dfs_path):
        if arquivo.endswith('.csv') == False:
            continue
        dados_estacao = pd.read_csv(os.path.join(dfs_path, arquivo)) 
        dados_estacao.set_index('Data',inplace=True)
        precipitation_df[arquivo[0:4]] = dados_estacao[coluna]

gera_df('Precipitacao')
precipitation_df.reset_index()
precipitation_df.to_csv('precipitacao.csv')
#quantidade_de_dias = precipitation_df.shape[0]
#print(precipitation_df.isnull().sum()/quantidade_de_dias)
#print(precipitation_df)