import pandas as pd
import os
import datetime

def ajeita_data_hora(string):
    data = string[0:10]
    hora = string[10:]
    is_UTC = False

    if data[4] == '-':
        data = datetime.datetime.strptime(data, r"%Y-%m-%d")
    elif data[4] == '/':
        data = datetime.datetime.strptime(data, r"%Y/%m/%d")
    
    if hora.find("UTC") != -1:
        hora = datetime.datetime.strptime(hora[:4],"%H%M")
        is_UTC = True
    else:    
        hora = datetime.datetime.strptime(hora,"%H:%M")

    hora = datetime.time(hora.hour,hora.minute)

    datahora = datetime.datetime.combine(data,hora)

    if is_UTC:
        datahora -= datetime.timedelta(hours=3) 
    return datahora

def concatena_csv(estacao): 
    folder_path = "C:\\Users\\Miguel\\Downloads\\ds\\DS\\por_estacao\\"+estacao
    colunas = ['data', 'hora', 'precipitacao', 'pressao_atmosferica', 'pressao_max', 'pressao_min', 'radiacao', 'temperatura_do_ar', 'temperatura_orvalho', 'temperatura_max', 'temperatura_min', 'temperatura_orvalho_max', 'temperatura_orvalho_min', 'umidade_max', 'umidade_min', 'umidade_relativa', 'direcao_vento', 'rajada_max', 'velocidade_media_vento']
    # Crie uma lista vazia para armazenar os dataframes lidos de cada arquivo csv
    dfs = []

    # Loop através de cada arquivo na pasta e ler o csv em um dataframe
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.CSV'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, encoding = "Latin-1", delimiter = ";", header=8)
            df.drop(df.columns[-1],axis = 1, inplace = True)
            df.columns = colunas

            df['data_hora'] = df['data'] + df['hora']

            df['data_hora'] = df['data_hora'].apply(ajeita_data_hora)

            df.drop(columns=['data','hora'],axis = 1, inplace = True)

            dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)
    big_df.replace('-9999', float('NaN'), inplace = True)
    big_df.replace(-9999, float('NaN'), inplace = True)
    big_df.replace(float(-9999), float('NaN'), inplace = True)

    colunas_selecionadas = big_df.columns[:-1]
    big_df[colunas_selecionadas] = big_df[colunas_selecionadas].replace(',', '.', regex=True).astype(float)
    big_df.to_csv('C:\\Users\\Miguel\\Downloads\\ds\\DS\\por_estacao\\concatenados\\'+estacao+'.CSV', index=False)
    print(estacao + " feita")


estacoes = ["A601","A602","A603","A604","A606","A607","A608","A609","A610","A611","A618","A619","A620","A621","A624","A625","A626","A627","A628","A629","A630","A635","A636","A652","A659","A667"]


for estacao in estacoes:    
    concatena_csv(estacao)