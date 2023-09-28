import pandas as pd
import os
import datetime
import xml.etree.ElementTree as ET

input_path = r'meteorologia\por_estacao'
output_path = r'meteorologia\concatenados_diarios'
#renomeador_de_colunas = {
#    DATA (YYYY-MM-DD): Data, 
#    HORA (UTC) : Hora, PRECIPITA��O TOTAL, HOR�RIO (mm);PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB);PRESS�O ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB);PRESS�O ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB);RADIACAO GLOBAL (KJ/m�);TEMPERATURA DO AR - BULBO SECO, HORARIA (�C);TEMPERATURA DO PONTO DE ORVALHO (�C);TEMPERATURA M�XIMA NA HORA ANT. (AUT) (�C);TEMPERATURA M�NIMA NA HORA ANT. (AUT) (�C);TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (�C);TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (�C);UMIDADE REL. MAX. NA HORA ANT. (AUT) (%);UMIDADE REL. MIN. NA HORA ANT. (AUT) (%);UMIDADE RELATIVA DO AR, HORARIA (%);VENTO, DIRE��O HORARIA (gr) (� (gr));VENTO, RAJADA MAXIMA (m/s);VENTO, VELOCIDADE HORARIA (m/s);
#}

def le_xlm():
    # Carrega o arquivo XML
    tree = ET.parse(r'meteorologia\concatenados_diarios\dicionario_de_colunas.xml')

    # Obtém a raiz do documento XML
    root = tree.getroot()

    # Cria um dicionário vazio
    dicionario = {}

    # Percorre os elementos do XML
    for elemento in root:
        # Obtém o atributo 'chave' de cada elemento
        chave = elemento.get('chave')
        # Obtém o texto do elemento como o valor correspondente
        valor = elemento.text
        # Adiciona a chave e o valor ao dicionário
        dicionario[chave] = valor

    # Exibe o dicionário resultante
    return(dicionario)

corretor = le_xlm()
corretor["\"UMIDADE RELATIVA DO AR HORARIA (%)\""] = 'Umidade_rel'

def ajeita_data_hora(string):
    data = string[0:10]
    hora = string[10:]

    if data[4] == '-':
        data = datetime.datetime.strptime(data, r"%Y-%m-%d")
    elif data[4] == '/':
        data = datetime.datetime.strptime(data, r"%Y/%m/%d")
    else: 
        print("Problema com horario")
    if hora.find("UTC") != -1:
        hora = datetime.datetime.strptime(hora[:4],"%H%M")
    else:    
        hora = datetime.datetime.strptime(hora,"%H:%M")

    hora = datetime.time(hora.hour,hora.minute)

    datahora = datetime.datetime.combine(data,hora)

    return datahora

def concatena_csv(estacao_path): 
    folder_path = os.path.join(input_path, estacao_path)
    colunas = corretor.values()    
    
    # Crie uma lista vazia para armazenar os dataframes lidos de cada arquivo csv
    dfs = []

    # Loop através de cada arquivo na pasta e ler o csv em um dataframe
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.CSV'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, encoding = "Latin-1", delimiter = ";", header=8)
            df.drop(df.columns[-1],axis = 1, inplace = True)
            df = df.rename(columns=corretor)

            df['Data_hora'] = df['Data'] + df['Hora']

            df['Data_hora'] = df['Data_hora'].apply(ajeita_data_hora)

            df.drop(columns=['Data','Hora'],axis = 1, inplace = True)

            dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)
    big_df.replace('-9999', float('NaN'), inplace = True)
    big_df.replace(-9999, float('NaN'), inplace = True)
    big_df.replace(float(-9999), float('NaN'), inplace = True)

    colunas_selecionadas = big_df.columns.tolist()
    colunas_selecionadas.remove('Data_hora')

    big_df[colunas_selecionadas] = big_df[colunas_selecionadas].replace(',', '.', regex=True).astype(float)
    caminho_com_extensao = os.path.join(output_path, estacao_path) + '.csv'
    big_df.to_csv(caminho_com_extensao, index=False)
    print(estacao_path + ' feita ')

for estacao_path in os.listdir(input_path):
    if estacao_path[0] == 'S':  
        concatena_csv(estacao_path)
