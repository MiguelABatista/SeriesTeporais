import pandas as pd
import os
import datetime

folder_path = r'meteorologia\concatenados_rio'
destino_path = r'meteorologia\diarios_rio'


def from_hourly_to_daily(estacao_path):
    data = pd.read_csv(os.path.join(folder_path, estacao_path))
    data['Data_hora'] = pd.to_datetime(data['Data_hora'])
    data = data.rename(columns={'Data_hora': 'Data'})
    data.set_index('Data', inplace=True)
    dados_diarios = data.resample('D').mean()
    dados_diarios.to_csv(os.path.join(destino_path, estacao_path))
    print(estacao_path + ' feita')
    return

for estacao_path in os.listdir(folder_path):
    from_hourly_to_daily(estacao_path)
