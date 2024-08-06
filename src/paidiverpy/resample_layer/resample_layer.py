""" Open raw image file
"""
import numpy as np
from paidiverpy import Paidiverpy
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt

class ResampleLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None,
                 logger=None,
                 images=None,
                 paidiverpy=None,
                 step_name=None,
                 parameters=None,
                 config_index=None,
                 raise_error=False,
                 verbose=True):

        super().__init__(config_file_path=config_file_path,
                         input_path=input_path,
                         output_path=output_path,
                         catalog_path=catalog_path,
                         catalog_type=catalog_type,
                         catalog=catalog,
                         config=config,
                         logger=logger,
                         images=images,
                         paidiverpy=paidiverpy,
                         raise_error=raise_error,
                         verbose=verbose)
        self.config_index = config_index
        self.step_name = step_name or 'sampling'
        if parameters:
            if not parameters.get('step_name'):
                parameters['step_name'] = self.step_name
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_order = len(self.images.steps)
        if self.step_order == 0:
            self.step_metadata = self._calculate_steps_metadata(self.config.general)
        else:
            self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])

    def run(self):
        test = self.step_metadata.get('test')
        limits = self.step_metadata.get('limits')
        mode = self.step_metadata.get('mode')
        if not mode:
            raise ValueError("Mode not defined")
        if mode == 'datetime':
            catalog = self._by_datetime(limits, self.step_order, test)
        elif mode == 'depth':
            catalog = self._by_depth(limits, self.step_order, test)
        elif mode == 'altitude':
            catalog = self._by_altitude(limits, self.step_order, test)
        elif mode == 'pitch_roll':
            catalog = self._by_pitch_roll(limits, self.step_order, test)
        elif mode == 'overlapping':
            catalog = self._by_overlapping(limits, self.step_order, test)
        elif mode == 'remove_obscure_photos':
            pass
        elif mode == 'fixed':
            catalog = self._by_fixed_number(limits, self.step_order, test)
        elif mode == 'percent':
            catalog = self._by_percent(limits, self.step_order, test)
        elif mode == 'region':
            catalog = self._by_region(limits, self.step_order, test)
        #     values_type = kwargs.get('values_type')
        #     catalog = self._by_limits(self.step_order, test, values, values_type)
        elif mode == 'obscure':
            catalog = self.remove_obscure_photos(limits, self.step_order, test)
        else:
            raise ValueError("Mode not supported.")
        if self.step_order ==  0:
            return catalog
        if not test:
            new_catalog = self.get_catalog(flag='all')
            new_catalog.loc[new_catalog.index.isin(catalog.index), 'flag'] = catalog.flag
            self.set_catalog(new_catalog)
            self.step_name = f'trim_{mode}' if not self.step_name else self.step_name
            self.images.add_step(f'trim_{mode}', catalog=self.get_catalog())

    def _by_percent(self, limits, step_order=None, test=False):
        catalog = self.get_catalog().sample(frac=1, random_state=np.random.seed())
        new_catalog = catalog.sample(frac=limits)
        if step_order == 0:
            return new_catalog
        if test:
            self.plot_trimmed_photos(new_catalog)
            return None
        catalog.loc[~catalog.index.isin(new_catalog.index), 'flag'] = step_order
        return catalog

    def _by_fixed_number(self, limits, step_order=None, test=False):
        catalog = self.get_catalog().sample(frac=1, random_state=np.random.seed())
        if limits > len(self.get_catalog()):
            self.logger.info("Number of photos to be removed is greater than the number of photos in the catalog.")
            self.logger.info("No photos will be removed.")
            return self.get_catalog()
        new_catalog = catalog.sample(n=limits)
        if step_order == 0:
            return new_catalog
        if test:
            self.plot_trimmed_photos(new_catalog)
            return None
        catalog.loc[~catalog.index.isin(new_catalog.index), 'flag'] = step_order
        return catalog

    def _by_datetime(self, limits, step_order=None, test=False):
        catalog = self.get_catalog()
        start_date, end_date = limits
        if not start_date and not end_date:
            return catalog
        if start_date is None:
            start_date = catalog['datetime'].min()
        else:
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = catalog['datetime'].max()
        else:
            end_date = pd.to_datetime(end_date)
        if start_date > end_date:
            raise ValueError("Start date cannot be greater than end date")
        if step_order == 0:
            return catalog.loc[(catalog['datetime'] >= start_date) & (catalog['datetime'] <= end_date)]
        catalog.loc[(catalog['datetime'] < start_date) | (catalog['datetime'] > end_date), 'flag'] = step_order
        self.logger.info("Number of photos to be removed: %s", catalog.flag[catalog.flag == step_order].count())
        if test:
            self.plot_trimmed_photos(catalog[catalog.flag == 0])
            return None
        return catalog

    def _by_depth(self, limits, step_order=None, test=False):
        catalog = self.get_catalog()
        catalog.loc[:, 'depth_m'] = catalog['depth_m'].abs()
        if limits[0] == 'lower':
            catalog.loc[catalog['depth_m'] < limits[1], 'flag'] = step_order
        else:
            catalog.loc[catalog['depth_m'] > limits[1], 'flag'] = step_order
        self.logger.info("Number of photos to be removed: %s", catalog.flag[catalog.flag == step_order].count())
        if test:
            self.plot_trimmed_photos(catalog[catalog.flag == 0])
            return None
        return catalog

    def _by_altitude(self, limits, step_order=None, test=False):
        catalog = self.get_catalog()
        catalog.loc[:, 'altitude_m'] = catalog['altitude_m'].abs()
        catalog.loc[catalog['altitude_m'] > limits, 'flag'] = step_order
        self.logger.info("Number of photos to be removed: %s", catalog.flag[catalog.flag == step_order].count())
        if test:
            self.plot_trimmed_photos(catalog[catalog.flag == 0])
            return None
        return catalog

    def _by_pitch_roll(self, limits, step_order=None, test=False):
        catalog = self.get_catalog()
        catalog.loc[:, 'pitch_deg'] = catalog['pitch_deg'].abs()
        catalog.loc[:, 'roll_deg'] = catalog['roll_deg'].abs()
        catalog.loc[(catalog['pitch_deg'] > limits[0]) & (catalog['roll_deg'] > limits[1]), 'flag'] = step_order
        self.logger.info("Number of photos to be removed: %s", catalog.flag[catalog.flag == step_order].count())
        if test:
            self.plot_trimmed_photos(catalog[catalog.flag == 0])
            return None
        return catalog

    def _by_region(self, limits, step_order=None, test=False): # values_type = 'limits', 'geojson', 'shapefile'
        catalog = self.get_catalog()
        if isinstance(limits, str):
            polygons = gpd.read_file(limits)
        else:
            min_lon, max_lon, min_lat, max_lat = limits
            polygons = gpd.GeoDataFrame(geometry=[Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)])])

        def point_in_any_polygon(point):
            return any(polygon.contains(point) for polygon in polygons.geometry)

        catalog['flag'] = catalog.apply(lambda x: x['flag'] if point_in_any_polygon(x['point']) else step_order, axis=1)
        self.logger.info("Number of photos to be removed: %s", catalog.flag[catalog.flag == step_order].count())
        if test:
            self.plot_trimmed_photos(catalog[catalog.flag == 0])
            return None
        return catalog

    def remove_obscure_photos(self, limits, step_order=None, test=False):
        catalog = self.get_catalog()
        images = self.images.get_step(step=self.config_index, by_order=True)
        brightnesss = []
        for index, img in enumerate(images):
            img_data = img.image
            brightness = np.mean(img_data)
            if brightness < limits[0] or brightness > limits[1]:
                catalog.loc[index, 'flag'] = step_order
            brightnesss.append(brightness)
        self.logger.info("Number of photos to be removed: %s", catalog.flag[catalog.flag == step_order].count())
        if test:
            self.plot_trimmed_photos(catalog[catalog.flag == 0])
            plt.hist(brightnesss, bins=30, edgecolor='black')
            plt.xlabel('Mean RGB Brightness')
            plt.ylabel('Frequency')
            plt.title('Distribution of Image Brightness')
            plt.show()
            return None
        return catalog

    def _by_overlapping(self, limits, step_order=None, test=False):
        catalog = self.get_catalog()
        catalog.loc[:, 'pitch_deg'] = catalog['pitch_deg'].abs()
        catalog.loc[:, 'roll_deg'] = catalog['roll_deg'].abs()

        theta = limits[0]
        omega = limits[1]
        overlap_threshold = limits[2]
        catalog['approx_vertdim_m'] = 2 * (catalog['altitude_m'] + 1.12) * np.tan(np.radians(theta / 2))
        catalog['approx_horizdim_m'] = 2 * (catalog['altitude_m'] + 1.12) * np.tan(np.radians(omega / 2))
        catalog['approx_area_m2'] = 4 * ((catalog['altitude_m'] + 1.12) ** 2) * np.tan(np.radians(theta / 2)) * np.tan(np.radians(omega / 2))
        catalog['headingoffset_rad'] = np.arctan(catalog['approx_horizdim_m'] / catalog['approx_vertdim_m'])
        catalog['cornerdist_m'] = 0.5 * catalog['approx_horizdim_m'] / np.sin(catalog['headingoffset_rad'])
        catalog['longpos_deg'] = catalog['lon'] + 360

        corner_columns = ['TRcornerlong', 'TRcornerlat', 'TLcornerlong', 'TLcornerlat', 'BLcornerlong', 'BLcornerlat', 'BRcornerlong', 'BRcornerlat']
        catalog[corner_columns] = 0.0
        # Iterate over each photo to calculate corner coordinates
        for i, row in catalog.iterrows():
            lat, lon, heading_deg, headingoffset_rad, cornerdist_m = row[['lat', 'longpos_deg', 'heading_deg', 'headingoffset_rad', 'cornerdist_m']]

            catalog.loc[i, 'TRcornerlong'], catalog.loc[i, 'TRcornerlat'] = ResampleLayer.calculate_corner(lat, lon, heading_deg, headingoffset_rad, cornerdist_m, 0)
            catalog.loc[i, 'TLcornerlong'], catalog.loc[i, 'TLcornerlat'] = ResampleLayer.calculate_corner(lat, lon, heading_deg, headingoffset_rad, cornerdist_m, -2 * headingoffset_rad * 180 / np.pi)
            catalog.loc[i, 'BLcornerlong'], catalog.loc[i, 'BLcornerlat'] = ResampleLayer.calculate_corner(lat, lon, heading_deg, headingoffset_rad, cornerdist_m, 180)
            catalog.loc[i, 'BRcornerlong'], catalog.loc[i, 'BRcornerlat'] = ResampleLayer.calculate_corner(lat, lon, heading_deg, headingoffset_rad, cornerdist_m, 180 - 2 * headingoffset_rad * 180 / np.pi)

        # Calculate the coordinates for the first photo
        n = pd.DataFrame({
            'long_deg': [catalog['TLcornerlong'].iloc[0], catalog['TRcornerlong'].iloc[0], catalog['BRcornerlong'].iloc[0], catalog['BLcornerlong'].iloc[0]],
            'lat_deg': [catalog['TLcornerlat'].iloc[0], catalog['TRcornerlat'].iloc[0], catalog['BRcornerlat'].iloc[0], catalog['BLcornerlat'].iloc[0]]
        })

        chn = np.append(Polygon(n).convex_hull.exterior.coords, [Polygon(n).convex_hull.exterior.coords[0]], axis=0)
        coordsn = pd.DataFrame(chn, columns=['long_deg', 'lat_deg'])

        # Find overlaps
        catalog['overlap'] = 0
        catalog['polygon_m'] = Polygon(coordsn.values)

        for i in range(1, len(catalog)):
            m = pd.DataFrame({
                'long_deg': [catalog['TLcornerlong'].iloc[i], catalog['TRcornerlong'].iloc[i], catalog['BRcornerlong'].iloc[i], catalog['BLcornerlong'].iloc[i]],
                'lat_deg': [catalog['TLcornerlat'].iloc[i], catalog['TRcornerlat'].iloc[i], catalog['BRcornerlat'].iloc[i], catalog['BLcornerlat'].iloc[i]]
            })

            chm = np.append(Polygon(m).convex_hull.exterior.coords, [Polygon(m).convex_hull.exterior.coords[0]], axis=0)
            coordsm = pd.DataFrame(chm, columns=['long_deg', 'lat_deg'])

            polygon_n = Polygon(coordsn.values)
            polygon_m = Polygon(coordsm.values)
            catalog.loc[i, 'polygon_m'] = polygon_m
            if polygon_n.intersects(polygon_m):
                if overlap_threshold is None:
                    catalog.loc[i, 'overlap'] = 1
                else:
                    overlap_area = polygon_n.intersection(polygon_m).area
                    overlap_percentage_n = overlap_area / polygon_n.area
                    overlap_percentage_m = overlap_area / polygon_m.area
                    if overlap_percentage_n > overlap_threshold or overlap_percentage_m > overlap_threshold:
                        catalog.loc[i, 'overlap'] = 1
                    else:
                        coordsn = coordsm
            else:
                coordsn = coordsm
        self.logger.info("Number of photos to be removed: %s", int(catalog['overlap'].sum()))
        new_catalog = self.get_catalog()
        
        new_catalog.loc[catalog['overlap'] == 1, 'flag'] = step_order
        if test:
            ResampleLayer.plot_polygons(catalog)
            self.plot_trimmed_photos(new_catalog[new_catalog.flag == 0])
            return None
        return new_catalog

    @staticmethod
    def plot_polygons(catalog):
        gdf = gpd.GeoDataFrame(catalog, geometry='polygon_m')
        _, ax = plt.subplots(figsize=(30, 30))

        gdf[gdf.overlap == 0].plot(ax=ax,facecolor='none', edgecolor='black', label='No Overlap')
        gdf[gdf.overlap == 1].plot(ax=ax,facecolor='none', edgecolor='red', label='Overlap')
        plt.savefig('overlap.png')
        plt.show()

    @staticmethod
    def calculate_corner(lat, lon, heading_deg, headingoffset_rad, cornerdist_m, angle_offset):
        angle = (headingoffset_rad * 180 / np.pi) + heading_deg + angle_offset
        destination = geodesic(meters=cornerdist_m).destination((lat, lon), angle)
        return destination.longitude, destination.latitude
