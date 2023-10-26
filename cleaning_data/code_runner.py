'''
This code is used to run all codes that do the first layer of data cleaning
For this you'll need to dowload all files from https://portal.inmet.gov.br/dadoshistoricos
And leave it (in the zip form) at the files\raw_files folder
'''

import getting_rio
import organizing_by_station
import concatenate

raw_file_path = r"C:\Users\miguel.batista_bigda\Documents\GitHub\SeriesTeporais\files\raw_files"
just_rio_path = r"C:\Users\miguel.batista_bigda\Documents\GitHub\SeriesTeporais\files\just_rio"
organized_by_station_path = r'C:\Users\miguel.batista_bigda\Documents\GitHub\SeriesTeporais\files\organized_by_station'
xlm_dict_path = r'C:\Users\miguel.batista_bigda\Documents\GitHub\SeriesTeporais\cleaning_data\column_dict.xml'
daily_data_path = r'C:\Users\miguel.batista_bigda\Documents\GitHub\SeriesTeporais\files\daily_data'

#First we'll get from all station, just the rio de janeiro stations
getting_rio.main(raw_file_path, just_rio_path)
print("Getting Rio done!")

#Then we'll organize all this stations in folders by station and not by year
organizing_by_station.main(just_rio_path, organized_by_station_path)
print("Organizing by station done!")

#Now all setup is done, we can do the big cleaning
#This code take each station folder, concatenate all files in a single dataframe
#Besides that it normalize the columns names, the NaN values and the date format
#For last it tranform the data from hourly to daily
concatenate.main(organized_by_station_path, xlm_dict_path, daily_data_path)
print("Concatenate done!")

