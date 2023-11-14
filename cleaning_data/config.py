import os

base_path = r"C:\Users\miguel.batista_bigda\Documents\GitHub\SeriesTeporais" # Path to the main folder
raw_file_path = os.path.join(base_path, "files", "raw_files") 
just_rio_path = os.path.join(base_path, "files", "just_rio")
organized_by_station_path = os.path.join(base_path, "files", "organized_by_station")
xlm_dict_path = os.path.join(base_path, "cleaning_data", "column_dict.xml")
daily_data_path = os.path.join(base_path, "files", "daily_data")
IDW_interpolation_path = os.path.join(base_path, "files", "IDW_interpolation")