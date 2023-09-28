import os
import shutil

# Diretório onde estão os arquivos meteorológicos
folder_origem = 'meteorologia\\arquivo_bruto\\'
folder_destino = r'meteorologia\por_estacao'
# Percorre todos os arquivos no diretório

for ano in range(2000,2023):
    diretorio_origem = folder_origem + str(ano) + "\\"
    for arquivo in os.listdir(diretorio_origem):
        if arquivo.endswith('.CSV'):  # Verifica se é um arquivo de texto (ou o formato correto)
            estacao = arquivo.split('_')[3]  # Obtém o código da estação do nome do arquivo
            if estacao[0] != 'A':
                print('PROBLEMA')
                print(arquivo)
            diretorio_destino = os.path.join(folder_destino, estacao)  # Caminho da pasta de destino
            os.makedirs(diretorio_destino, exist_ok=True)  # Cria a pasta se não existir
            caminho_origem = os.path.join(diretorio_origem, arquivo)  # Caminho completo do arquivo de origem
            caminho_destino = os.path.join(diretorio_destino, arquivo)  # Caminho completo do arquivo de destino
            shutil.move(caminho_origem, caminho_destino)  # Move o arquivo para a pasta de destino
