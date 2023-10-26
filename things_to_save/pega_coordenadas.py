import pandas as pd
import os
import csv

str = 'INMET_NE_BA_A401_SALVADOR_01-01-2022_A_31-12-2022.CSV'
raw_df_path = r'C:\Users\Miguel\Documents\GitHub\SeriesTeporais\meteorologia\2022'

def is_in_rio(file):
    splited_file = file.split('_')
    station = splited_file[3]

    try:
        station_number = int(station[1:])
    except:
        print('ERRO')
        print(file)
        print()
        return False
    
    if station_number > 600 and station_number < 700:
        return True
    return False

def get_coordinates(file):
    # Abre o arquivo CSV
    with open(os.path.join(raw_df_path,file), newline='', encoding='Latin-1') as csvfile:
        # Lê o conteúdo do arquivo CSV
        csvreader = csv.reader(csvfile, delimiter=';')
        
        # Itera pelas linhas do arquivo
        for row in csvreader:
            # Procura as linhas que contêm latitude e longitude
            if row[0] == 'LATITUDE:':
                latitude = float(row[1].replace(',', '.'))
            elif row[0] == 'LONGITUDE:':
                longitude = float(row[1].replace(',', '.'))

    return (latitude,longitude)


estacoes = []
latitudes = []
longitudes = []
df = pd.DataFrame()

for file in os.listdir(raw_df_path):
    if file.endswith('.CSV') == False:
        continue
    if is_in_rio(file) == False:
        continue
    latitude, longitude = get_coordinates(file)
    station= file.split('_')[3]
    df[station] = [latitude, longitude]


print(df)
df.to_csv('dados_geograficos.csv', index=False)


