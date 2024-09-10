import glob
from pathlib import Path
from typing import Dict
import yaml
import pandas as pd
import mariqt.sources.ifdo as miqtifdo
import mariqt.tests as miqtt
from mariqt.core import IfdoException
from shapely.geometry import Point

from paidiverpy.config import Configuration
from utils import initialise_logging

filename_columns = ["filename", "file_name", "FileName", "File Name"]
index_columns = ["id", "index", "ID", "Index", "Id"]
datetime_columns = ["datetime", "date_time", "DateTime", "Datetime"]
lat_columns = ["lat", "latitude_deg", "latitude", "Latitude", "Latitude_deg", "Lat"]
lon_columns = ["lon", "longitude_deg", "longitude", "Longitude", "Longitude_deg", "Lon"]


class CatalogParser:
    def __init__(
        self,
        config=None,
        catalog_path=None,
        catalog_type=None,
        append_data_to_catalog=None,
        logger=None,
    ):

        self.logger = logger or initialise_logging()
        self.config = config or self._build_config(
            catalog_path, catalog_type, append_data_to_catalog
        )
        self.catalog_type = getattr(self.config.general, "catalog_type", None)
        self.append_data_to_catalog = getattr(
            self.config.general, "append_data_to_catalog", None
        )
        self.catalog_path = getattr(self.config.general, "catalog_path", None)
        if not self.catalog_path:
            raise ValueError("Catalog path is not specified.")
        if not self.catalog_type:
            raise ValueError("Catalog type is not specified.")

        self.catalog = self.open_catalog()

    def _build_config(self, catalog_path, catalog_type, append_data_to_catalog):
        general_params = {
            "catalog_path": catalog_path,
            "catalog_type": catalog_type,
            "append_data_to_catalog": append_data_to_catalog,
        }
        config = Configuration()
        config.add_config("general", general_params)
        return config

    def open_catalog(self):
        if self.catalog_type == "IFDO":
            catalog = self._open_ifdo_catalog()
        elif self.catalog_type == "CSV":
            catalog = self._open_csv_catalog()
        else:
            raise ValueError("Catalog type is not supported.")

        if self.append_data_to_catalog:
            catalog = self._add_data_to_catalog(catalog)

        catalog["flag"] = 0
        catalog = self._process_coordinates(catalog)
        return catalog

    def _process_coordinates(self, catalog):

        catalog = self._rename_columns(catalog, lat_columns)
        catalog = self._rename_columns(catalog, lon_columns)
        if "lon" in catalog.columns and "lat" in catalog.columns:
            catalog["point"] = catalog.apply(
                lambda x: Point(x["lon"], x["lat"]), axis=1
            )

        return catalog

    def _rename_columns(self, catalog, columns, raise_error=False):
        if not any(col in catalog.columns for col in columns):
            if raise_error:
                self.logger.error(
                    "Catalog does not have a %s type column. It should have one of the following columns: %s",
                    columns[0],
                    columns,
                )
                raise ValueError(
                    f"Catalog does not have a {columns[0]} type column. It should have one of the following columns: {columns}"
                )
            self.logger.warning(
                "Catalog does not have a %s type column. It should have one of the following columns: %s",
                columns[0],
                columns,
            )
            self.logger.warning("Some functions may not work properly.")

            return catalog

        for col in columns:
            if col in catalog.columns:
                catalog.rename(columns={col: columns[0]}, inplace=True)
                columns_1 = columns.copy()
                columns_1.remove(columns_1[0])
                catalog.drop(columns_1, errors="ignore", inplace=True)
                return catalog

    def _add_data_to_catalog(self, catalog):
        new_catalog = pd.read_csv(self.append_data_to_catalog).drop_duplicates(
            subset="filename", keep="first"
        )

        if not any(col in new_catalog.columns for col in filename_columns):
            raise ValueError(
                "Catalog does not have a filename column. It should have one of the following columns: 'filename', 'file_name', 'FileName', 'File Name'."
            )

        new_catalog = self._rename_columns(new_catalog, filename_columns)
        catalog = catalog.merge(new_catalog, how="left", on="filename")

        return catalog

    def _open_ifdo_catalog(self):
        catalog = miqtifdo.iFDO_Reader(self.catalog_path).ifdo
        self._validate_ifdo(catalog)
        return catalog

    def _open_csv_catalog(self):
        catalog = pd.read_csv(self.catalog_path)

        if not any(col in catalog.columns for col in index_columns):
            catalog = catalog.reset_index().rename(columns={"index": "ID"})

        catalog = self._rename_columns(catalog, filename_columns, raise_error=True)
        catalog = self._rename_columns(catalog, datetime_columns)
        if "datetime" in catalog.columns:
            catalog["datetime"] = pd.to_datetime(catalog["datetime"])
            catalog.sort_values(by="datetime", inplace=True)

        return catalog

    @staticmethod
    def _validate_ifdo(ifdo_data: Dict):
        """Validates input data against iFDO scheme. Found errors are printed to the terminal.

        Args:
            ifdo_data (Dict): parsed iFDO data.
        """
        miqtt.are_valid_ifdo_fields(ifdo_data["image-set-header"])
        unique_names = miqtt.filesHaveUniqueName(ifdo_data["image-set-items"].keys())
        if unique_names:
            raise IfdoException(
                {"Validation error": f"Duplicate filenames found: {unique_names}"}
            )

    def __repr__(self) -> str:
        # Return the string representation of the catalog DataFrame
        return repr(self.catalog)

    def _repr_html_(self) -> str:
        message = "This is a instance of 'CatalogParser'<br><br>"

        # Return the HTML representation of the catalog DataFrame for Jupyter
        return message + self.catalog._repr_html_()

    # def load_waypoints(self):
    #     waypoint_folder_path = Path(self.config.position.waypoint_folder_path)
    #     file_pattern = str(waypoint_folder_path.joinpath(self.config.position.waypoint_data_pattern))
    #     list_of_files = glob.glob(file_pattern)
    #     for file in list_of_files:
    #         with open(file, "r", encoding='utf-8') as yaml_file:
    #             waypt_list = yaml.safe_load(yaml_file)
    #         # Extract names, types, positions, and attitudes of waypoints into a DataFrame
    #         waypoints = pd.DataFrame({
    #             'name': [item['name'] for item in waypt_list['mission_items']],
    #             'type': [item['type'] for item in waypt_list['mission_items']],
    #             'position': [item['position'] for item in waypt_list['mission_items']],
    #             'attitude': [item['attitude'] for item in waypt_list['mission_items']],
    #             'datetime': [item['created'] if item.get('created') else None for item in waypt_list['mission_items']]
    #         })
    #         waypoints['longitude_deg'] = waypoints['position'].apply(lambda x: x.split()[0])
    #         waypoints['latitude_deg'] = waypoints['position'].apply(lambda x: x.split()[1])
    #     waypoints.dropna(inplace=True)
    #     waypoints['datetime'] = pd.to_datetime(waypoints['datetime'], unit='s')
    #     waypoints.sort_values(by='datetime', inplace=True)
    #     self.catalog = self.merge_waypoints_to_catalog(waypoints)
    #     return waypoints

    # def merge_waypoints_to_catalog(self, waypoints):
    #     transect = []
    #     catalog = self.catalog.copy()

    #     # Iterate through each photo
    #     for _, photo_row in catalog.iterrows():
    #         photo_time = photo_row['datetime']
    #         photo_find = False
    #         # Find the transect by checking the intervals
    #         for i in range(len(waypoints) - 1):
    #             start_time = waypoints.iloc[i]['datetime']
    #             end_time = waypoints.iloc[i + 1]['datetime']

    #             if start_time <= photo_time < end_time:
    #                 photo_find = True
    #                 transect.append(i+1)
    #                 break
    #         if not photo_find:
    #             transect.append(0)
    #     catalog['transect'] = transect
    #     return catalog
