""" Open raw image file
"""
import shutil
import os

import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from PIL import Image

from paidiverpy import Paidiverpy

class SizeLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None):
        super().__init__(config_file_path, input_path, output_path, catalog_path, catalog_type, catalog, config)

    def trim_by_transect(self, filenames=None):
        # Convert time to format in photo names
        scidata_df = pd.read_csv(filenames[0], sep=";")
        scidata_df['phototime'] = pd.to_datetime(scidata_df['time_unix_seconds'], unit='s', origin='unix')
        scidata_df['photoname'] = scidata_df['phototime'].dt.strftime('%Y%m%d%H%M%S')

        # Trimmed science data did not include photos, so create science data files from raw
        data_frames = [pd.read_csv(file, skiprows=1) for file in filenames[1:]]
        scidata_combined = pd.concat(data_frames, ignore_index=True)

        scidata_combined['longitude_deg'] = scidata_combined['longitude_deg'].apply(lambda x: -x if x > 0 else x)
        scidata_combined['phototime'] = pd.to_datetime(scidata_combined['time_unix_seconds'], unit='s', origin='unix')
        scidata_combined['photoname'] = scidata_combined['phototime'].dt.strftime('%Y%m%d%H%M%S')

        scidata_combined.to_csv('Science_data_20180525_all_raw.csv', index=False)

        scidata_combined = scidata_combined[scidata_combined['altitude_m'] <= 6000]

        # Define transects
        transects = [
            (20180525072300, 20180525090300, 'KM25_T1'),
            (20180525113000, 20180525123600, 'KM25_T2'),
            (20180526145400, 20180526150100, 'KM29_T1'),
            (20180527125700, 20180527153000, 'KM33_T1'),
            (20180528205300, 20180528235600, 'KM36_T1'),
            (20180529001900, 20180529021000, 'KM36_T2'),
            (20180601143500, 20180601161500, 'KM43_T1'),
            (20180602175400, 20180602192200, 'KM47_T1')
        ]

        for start, end, transect_name in transects:
            transect_data = scidata_combined[(scidata_combined['photoname'].astype(int) > start) & 
                                             (scidata_combined['photoname'].astype(int) < end)]
            transect_unique = transect_data.drop_duplicates(subset='photoname')

            all_photos = pd.DataFrame({
                'photoname': [os.path.splitext(photo)[0] for photo in os.listdir(f"/Volumes/DeepCCZ1 Drobo/Jen's working files/Still photos renamed w timestamp/{transect_name[:5]}")]
            })

            transect_photos = all_photos[(all_photos['photoname'].astype(int) > start) & 
                                         (all_photos['photoname'].astype(int) < end)]

            transect_scidata = transect_unique[transect_unique['photoname'].isin(transect_photos['photoname'])]

            old_folder = f"{transect_name[:5]}"
            new_folder = f"{transect_name}"
            os.makedirs(new_folder, exist_ok=True)

            for photo_name in transect_photos['photoname']:
                shutil.copy(os.path.join(old_folder, f"{photo_name}.JPG"),
                            os.path.join(new_folder, f"{photo_name}.JPG"))


    def calculate_approx_dimensions(self, altitude, theta=40, omega=57):
        altitude += 1.12
        approx_vertdim_m = 2 * altitude * np.tan(np.radians(theta) / 2)
        approx_horizdim_m = 2 * altitude * np.tan(np.radians(omega) / 2)
        approx_area_m2 = 4 * (altitude ** 2) * np.tan(np.radians(theta) / 2) * np.tan(np.radians(omega) / 2)
        return approx_vertdim_m, approx_horizdim_m, approx_area_m2

    def find_corners(self, longitude, latitude, heading, vertdim, horizdim):
        headingoffset_rad = np.arctan(horizdim / vertdim)
        cornerdist_m = 0.5 * horizdim / np.sin(headingoffset_rad)

        corners = []
        for angle_offset in [headingoffset_rad, -headingoffset_rad, np.pi - headingoffset_rad, np.pi + headingoffset_rad]:
            angle = np.degrees(heading) + np.degrees(angle_offset)
            point = geodesic(meters=cornerdist_m).destination(Point(latitude, longitude), angle)
            corners.append((point.longitude, point.latitude))
        return corners

    def trim_overlapping(self):
        data = self.photo_data
        data['approx_vertdim_m'], data['approx_horizdim_m'], data['approx_area_m2'] = zip(*data['altitude_m'].apply(self.calculate_approx_dimensions))

        data['corners'] = data.apply(lambda row: self.find_corners(
            row['longitude_deg'], row['latitude_deg'], row['heading_deg'],
            row['approx_vertdim_m'], row['approx_horizdim_m']), axis=1)

        data['polygon'] = data['corners'].apply(Polygon)
        overlap_mask = np.zeros(len(data), dtype=bool)

        for i in range(1, len(data)):
            current_polygon = data.at[i, 'polygon']
            for j in range(max(0, i-6), i):
                if not overlap_mask[j] and data.at[j, 'polygon'].intersects(current_polygon):
                    overlap_mask[i] = True
                    break

        data['overlap'] = overlap_mask.astype(int)
        num_photos_removed = data['overlap'].sum()
        num_photos_retained = len(data) - num_photos_removed

        print(f"Number of photos to be removed: {num_photos_removed}")
        print(f"Number of photos to be copied: {num_photos_retained}")

        fig, ax = plt.subplots()
        for i, row in data.iterrows():
            polygon = row['polygon']
            x, y = polygon.exterior.xy
            ax.plot(x, y, 'r-' if row['overlap'] else 'b-')
        plt.show()

        return data

    def crop_images(self, width=6000, height=3000):
        # Ensure the output directory exists
        if not os.path.exists(self.config.general.output_dir):
            os.makedirs(self.config.general.output_dir)

        # List all image files in the input directory
        image_files = [f for f in os.listdir(self.config.general.input_dir) if os.path.isfile(os.path.join(self.config.general.input_dir, f))]

        for image_file in image_files:
            # Read the image
            image_path = os.path.join(self.config.general.input_dir, image_file)
            with Image.open(image_path) as img:
                # Perform the crop operation
                cropped_img = img.crop((0, 0, width, height))
                
                # Save the cropped image to the output directory
                output_path = os.path.join(self.config.general.output_dir, image_file)
                cropped_img.save(output_path, format="JPEG")