import pandas as pd
import os
import datetime
import xml.etree.ElementTree as ET
import config

def read_xlm(xlm_path):
    '''
    This function reads the XML file and returns a dictionary with the column names
    It is used to rename the columns of the CSV files because some columns have diferent names in diferent files
    '''
    # Load the XML file
    tree = ET.parse(xlm_path)

    # Get the root of the XML document
    root = tree.getroot()

    dictionary = {}

    # Iterate through the elements of the XML
    for element in root:
        key = element.get('chave')
        value = element.text
        dictionary[key] = value

    return dictionary

def fix_date_time(string):
    '''	
    This function fixes the date and time format,
    because the format is different in some files
    '''
    date = string[0:10]
    time = string[10:]

    if date[4] == '-':
        date = datetime.datetime.strptime(date, r"%Y-%m-%d")
    elif date[4] == '/':
        date = datetime.datetime.strptime(date, r"%Y/%m/%d")
    else: 
        print("Problem with date format")
    if time.find("UTC") != -1:
        time = datetime.datetime.strptime(time[:4],"%H%M")
    else:    
        time = datetime.datetime.strptime(time,"%H:%M")

    time = datetime.time(time.hour, time.minute)

    date_time = datetime.datetime.combine(date, time)

    return date_time

def concatenate_csv(station_path, input_path, output_path, corrector): 
    '''
    This function concatenates all CSV files from a station into a single CSV file
    Input parameters:
        station_path: the name of the station
        input_path: the path of the folder that contains the CSV files
        output_path: the path of the folder that will contain the concatenated CSV file
        corrector: a dictionary with the column names
    No output parameters
    '''

    folder_path = os.path.join(input_path, station_path)
    
    # Create an empty list to store the dataframes read from each CSV file
    dfs = []

    # Loop through each file in the folder and read the CSV into a dataframe
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.CSV'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, encoding="Latin-1", delimiter=";", header=8)

            # dropping a column that doesn't have a name
            df.drop(df.columns[-1], axis=1, inplace=True)
            df = df.rename(columns=corrector)
            df['Data_hora'] = df['Data'] + df['Hora']

            df['Data_hora'] = df['Data_hora'].apply(fix_date_time)

            df.drop(columns=['Data', 'Hora'], axis=1, inplace=True)

            dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)

    # Replace the missing values with NaN
    big_df.replace('-9999', float('NaN'), inplace=True)
    big_df.replace(-9999, float('NaN'), inplace=True)
    big_df.replace(float(-9999), float('NaN'), inplace=True)

    selected_columns = big_df.columns.tolist()
    selected_columns.remove('Data_hora')

    big_df[selected_columns] = big_df[selected_columns].replace(',', '.', regex=True).astype(float)
    path_with_extension = os.path.join(output_path, station_path) + '.csv'
    daily_df = from_hourly_to_daily(big_df)
    daily_df.to_csv(path_with_extension, index=True)
    print(station_path + ' done ')

def from_hourly_to_daily(df):
    '''
    This function transforms the hourly data into daily data by getting the mean of each day
    '''
    df['Data_hora'] = pd.to_datetime(df['Data_hora'])
    df = df.rename(columns={'Data_hora': 'Data'})
    df.set_index('Data', inplace=True)
    daily_df = df.resample('D').mean()
    return daily_df

def main():
    '''
    This function concatenates all CSV files from all stations into a single CSV file
    And this CSV file contains the daily data
    Input parameters:
        xlm_dict_path: the path of the XML file that contains the column names
        organized_by_station_path: the path of the folder that contains the CSV files
        daily_data_path: the path of the folder that will contain the concatenated CSV file
    No output parameters
    '''

    if not os.path.exists(config.daily_data_path):
        os.makedirs(config.daily_data_path)

    corrector = read_xlm(config.xlm_dict_path)
    #I wasnt able to read the columns with the special characters, so I had to do this
    corrector["\"UMIDADE RELATIVA DO AR HORARIA (%)\""] = 'Umidade_rel'

    for estacao_path in os.listdir(config.organized_by_station_path):
        if estacao_path[0] == 'A':  
            concatenate_csv(estacao_path, config.organized_by_station_path, config.daily_data_path, corrector)

if __name__ == '__main__':
    main()