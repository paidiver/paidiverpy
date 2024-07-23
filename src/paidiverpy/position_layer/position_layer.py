""" Open raw image file
"""
import os
import shutil
from datetime import datetime
import glob
from pathlib import Path
import yaml
import pandas as pd
from paidiverpy import Paidiverpy

class PositionLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None):
        super().__init__(config_file_path, input_path, output_path, catalog_path, catalog_type, catalog, config)
        self.waypoint_data = []

    def load_waypoints(self, save_output=False):
        waypoint_folder_path = Path(self.config.position.waypoint_folder_path)
        file_pattern = str(waypoint_folder_path.joinpath(self.config.position.waypoint_data_pattern))
        list_of_files = glob.glob(file_pattern)
        # Read the 12th YAML file in the list
        for file in list_of_files:
            waypt_list = yaml.safe_load(file)
            # Extract names, types, positions, and attitudes of waypoints into a DataFrame
            self.waypoint_data = pd.DataFrame({
                'Waypt.Name': [item['name'] for item in waypt_list['mission_items']],
                'type': [item['type'] for item in waypt_list['mission_items']],
                'ROV.position': [item['position'] for item in waypt_list['mission_items']],
                'ROV.attitude': [item['attitude'] for item in waypt_list['mission_items']]
            })

            self.waypoint_data['Lat.decdeg'] = self.waypoint_data['ROV.position'].apply(lambda x: x.split()[0])
            self.waypoint_data['Long.decdeg'] = self.waypoint_data['ROV.position'].apply(lambda x: x.split()[1])

            if save_output:
                self.waypoint_data.to_csv(self.output_path / 'waypoint_data.csv', index=False)

    def image_positioning(self, filenames=None):
        file_names = filenames if filenames else self.catalog['filename']
        img_path_list = [self.input_path / filename for filename in file_names]

        rov_scidata = load_csv('path_to_your_csv.csv')
        KM1808_25_ROV_scidata = convert_time_unix_to_posix(KM1808_25_ROV_scidata, 'time_unix_seconds')
        KM1808_25_ROV_scidata = format_photoname(KM1808_25_ROV_scidata)

        def load_csv(filepath, delimiter=";"):
            return pd.read_csv(filepath, delimiter=delimiter)

        def convert_time_unix_to_posix(df, time_column):
            df['phototime'] = pd.to_datetime(df[time_column], unit='s', origin='unix').dt.tz_localize('GMT')
            return df

        def format_photoname(df):
            df['photoname'] = df['phototime'].dt.strftime('%Y%m%d%H%M%S')
            return df

        def bind_dataframes(*args):
            return pd.concat(args, ignore_index=True)

        def plot_data(df, longitude_col, latitude_col, waypoints_lat, waypoints_long):
            import matplotlib.pyplot as plt
            plt.scatter(df[longitude_col], df[latitude_col], label="Data")
            plt.scatter(waypoints_lat, waypoints_long, color='red', label="Waypoints")
            plt.legend()
            plt.show()

        def rename_photos(df, start, end, photo_dir, new_dir):
            df = df[(df['photoname'] > start) & (df['photoname'] < end)]
            photo_list = df['photoname'].unique()
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for photo in photo_list:
                old_path = os.path.join(photo_dir, f"{photo}.JPG")
                new_path = os.path.join(new_dir, f"{photo}.JPG")
                if os.path.exists(old_path):
                    shutil.copy2(old_path, new_path)
            return photo_list

        scidata_20180525 = bind_dataframes(
            pd.read_csv('path_to_your_csv1.csv', skiprows=1),
            pd.read_csv('path_to_your_csv2.csv', skiprows=1),
            pd.read_csv('path_to_your_csv3.csv', skiprows=1),
            pd.read_csv('path_to_your_csv4.csv', skiprows=1)
        )

        scidata_20180525.loc[scidata_20180525['longitude_deg'] > 0, 'longitude_deg'] *= -1

        scidata_20180525 = convert_time_unix_to_posix(scidata_20180525, 'time_unix_seconds')
        scidata_20180525 = format_photoname(scidata_20180525)
        scidata_20180525 = scidata_20180525[scidata_20180525['altitude_m'] <= 6000]

        transect_waypts = load_csv('path_to_waypoints.csv')

        start_KM25_T1 = '20180525072300'
        end_KM25_T1 = '20180525090300'

        KM25_T1_allscidata = scidata_20180525[(scidata_20180525['photoname'] > start_KM25_T1) & (scidata_20180525['photoname'] < end_KM25_T1)]
        KM25_T1 = KM25_T1_allscidata.drop_duplicates(subset='photoname')

        plot_data(KM25_T1, 'longitude_deg', 'latitude_deg', transect_waypts['Lat.decdeg'][0:2], transect_waypts['Long.decdeg'][0:2])

        all_photos_25 = pd.DataFrame(os.listdir('/Volumes/DeepCCZ1 Drobo/Jen\'s working files/Still photos renamed w timestamp/KM1808-25'), columns=['photoname'])
        all_photos_25['photoname'] = all_photos_25['photoname'].str.replace('JPG', '').str.replace('.', '')

        KM25_T1_photos = all_photos_25[(all_photos_25['photoname'] > start_KM25_T1) & (all_photos_25['photoname'] < end_KM25_T1)]
        KM25_T1_scidata = KM25_T1[KM25_T1['photoname'].isin(KM25_T1_photos['photoname'])]

        old_folder = '/Volumes/DeepCCZ1 Drobo/Jen\'s working files/Still photos renamed w timestamp/KM1808-25'
        new_folder = '/Volumes/DeepCCZ1 Drobo/Jen\'s working files/Still photos transects - all/KM1808-25_T1'
        rename_photos(KM25_T1_photos, start_KM25_T1, end_KM25_T1, old_folder, new_folder)
