import os
import shutil
import zipfile

def extract_csv_from_zip(zip_file_path, destination_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith('.csv') and os.path.basename(file_name).startswith('A6'):
                # Extrai o arquivo para uma pasta temporária
                zip_ref.extract(file_name, destination_folder)

                # Copia o arquivo para a pasta de destino
                shutil.copy(os.path.join(destination_folder, file_name), r'\concatenados_rio')


# Exemplo de uso
zip_file_path = r'meteorologia\concatenados_diarios.zip'
destination_folder = r'meteorologia'
extract_csv_from_zip(zip_file_path, destination_folder)
