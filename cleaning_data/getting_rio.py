import os
import zipfile
import shutil
import config

def merge_folders(root_dir):
    '''
    Some zip files have a "double folder", like this:
    just_rio/2000/2000/ (all files are here)
    This function solves this problem by merging the two folders
    In the end, the folder structure will be like this:
    just_rio/2000/ (all files are here)
    Input parameters:
        root_dir: the root directory
    No output parameters
    '''

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if os.path.exists(os.path.join(full_path, dirname)):
                source_folder = os.path.join(full_path, dirname)
                destination_folder = os.path.join(root_dir, dirname)
                for item in os.listdir(source_folder):
                    item_path = os.path.join(source_folder, item)
                    if os.path.isfile(item_path):
                        shutil.move(item_path, destination_folder)
                os.rmdir(source_folder)

def main():
    '''
    This function extracts all files from the zip files that are in the folder "raw_files"
    And get just the stations that are in rio de janeiro
    The files are extracted to the folder "just_rio"
    in the end, the folder structure will be like this:
    just_rio/year/(all files are here)
    '''

    for year in range(2000, 2023):
        zip_file_path = os.path.join(config.raw_file_path, f"{year}.zip")
        
        #Create the folder (if it doesn't exist)
        year_output_dir = os.path.join(config.just_rio_path, str(year))
        os.makedirs(year_output_dir, exist_ok=True)
        
        # Loops over all zip files
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

            for file_name in zip_ref.namelist():
                if '_RJ_' in file_name:
                    zip_ref.extract(file_name, year_output_dir)

    merge_folders(config.just_rio_path)

if __name__ == "__main__":
    main()        
