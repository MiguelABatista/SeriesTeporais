import pandas as pd
import os
import datetime
import xml.etree.ElementTree as ET




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
colunas = corretor.values()
novas_colunas = []
for estacao_path in os.listdir(r'meteorologia\concatenados_diarios'):
    file_path = os.path.join(r'meteorologia\concatenados_diarios', estacao_path)
    if(file_path.endswith('.py') or file_path.endswith('.xml')):
        pass
    else:
        with open(file_path, 'r') as arquivo: 
            lista = arquivo.readline().split(',')
            for coluna in lista:
                if coluna not in colunas and coluna not in novas_colunas:
                    novas_colunas.append(coluna) 
    print(estacao_path + ' feito')
print(novas_colunas)


def f():
    for estacao_path in os.listdir(r'meteorologia\concatenados_diarios'):
        df = pd.read_csv(os.path.join(file_path, estacao_path))    
        for coluna in df.columns:
            if coluna not in colunas and coluna not in novas_colunas:
                novas_colunas.append(coluna)
        print(estacao_path + ' feito')
    return ()
