import os
import shutil

def main(just_rio_path, organized_by_station_path):
    '''
    This function organizes that are organized by year and puts them in a folder organized by station
    '''
    # Getting the list of years
    years = [str(i) for i in range(2000, 2023)]

    # For each year
    for year in years:
        # Year directory
        year_path = os.path.join(just_rio_path, year)

        # For each CSV file in the year
        for csv_file in os.listdir(year_path):
            if csv_file.endswith(".CSV"):
                # Getting the station code
                station_code = csv_file.split("_")[3]

                # Destination directory for the station
                station_destination = os.path.join(organized_by_station_path, station_code)

                # Creating the station directory if it doesn't exist
                if not os.path.exists(station_destination):
                    os.makedirs(station_destination)

                # Moving the CSV file to the station directory
                shutil.move(os.path.join(year_path, csv_file), os.path.join(station_destination, csv_file))

if __name__ == "__main__":
    just_rio_path = r'files\just_rio'
    organized_by_station_path = r'files\organized_by_station'
    main(just_rio_path, organized_by_station_path)
    print("Done!")
