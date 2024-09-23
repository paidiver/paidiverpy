import glob
from pathlib import Path
from typing import Dict
import yaml
import pandas as pd
import mariqt.sources.ifdo as miqtifdo
import mariqt.tests as miqtt
from mariqt.core import IfdoException
from shapely.geometry import Point

from paidiverpy.config.config import Configuration
from utils import initialise_logging

filename_columns = ["filename", "file_name", "FileName", "File Name"]
index_columns = ["id", "index", "ID", "Index", "Id"]
datetime_columns = ["datetime", "date_time", "DateTime", "Datetime"]
lat_columns = ["lat", "latitude_deg", "latitude", "Latitude", "Latitude_deg", "Lat"]
lon_columns = ["lon", "longitude_deg", "longitude", "Longitude", "Longitude_deg", "Lon"]


class MetadataParser:
    def __init__(
        self,
        config=None,
        metadata_path=None,
        metadata_type=None,
        append_data_to_metadata=None,
        logger=None,
    ):

        self.logger = logger or initialise_logging()
        self.config = config or self._build_config(
            metadata_path, metadata_type, append_data_to_metadata
        )
        self.metadata_type = getattr(self.config.general, "metadata_type", None)
        self.append_data_to_metadata = getattr(
            self.config.general, "append_data_to_metadata", None
        )
        self.metadata_path = getattr(self.config.general, "metadata_path", None)
        if not self.metadata_path:
            raise ValueError("Metadata path is not specified.")
        if not self.metadata_type:
            raise ValueError("Metadata type is not specified.")

        self.metadata = self.open_metadata()

    def _build_config(self, metadata_path, metadata_type, append_data_to_metadata):
        general_params = {
            "metadata_path": metadata_path,
            "metadata_type": metadata_type,
            "append_data_to_metadata": append_data_to_metadata,
        }
        config = Configuration()
        config.add_config("general", general_params)
        return config

    def open_metadata(self):
        if self.metadata_type == "IFDO":
            metadata = self._open_ifdo_metadata()
        elif self.metadata_type == "CSV_FILE":
            metadata = self._open_csv_metadata()
        else:
            raise ValueError("Metadata type is not supported.")

        if self.append_data_to_metadata:
            metadata = self._add_data_to_metadata(metadata)

        metadata["flag"] = 0
        metadata = self._process_coordinates(metadata)
        return metadata

    def _process_coordinates(self, metadata):

        metadata = self._rename_columns(metadata, lat_columns)
        metadata = self._rename_columns(metadata, lon_columns)
        if "lon" in metadata.columns and "lat" in metadata.columns:
            metadata["point"] = metadata.apply(
                lambda x: Point(x["lon"], x["lat"]), axis=1
            )

        return metadata

    def _rename_columns(self, metadata, columns, raise_error=False):
        if not any(col in metadata.columns for col in columns):
            if raise_error:
                self.logger.error(
                    "Metadata does not have a %s type column. It should have one of the following columns: %s",
                    columns[0],
                    columns,
                )
                raise ValueError(
                    f"Metadata does not have a {columns[0]} type column. It should have one of the following columns: {columns}"
                )
            self.logger.warning(
                "Metadata does not have a %s type column. It should have one of the following columns: %s",
                columns[0],
                columns,
            )
            self.logger.warning("Some functions may not work properly.")

            return metadata

        for col in columns:
            if col in metadata.columns:
                metadata.rename(columns={col: columns[0]}, inplace=True)
                columns_1 = columns.copy()
                columns_1.remove(columns_1[0])
                metadata.drop(columns_1, errors="ignore", inplace=True)
                return metadata

    def _add_data_to_metadata(self, metadata):
        new_metadata = pd.read_csv(self.append_data_to_metadata).drop_duplicates(
            subset="filename", keep="first"
        )

        if not any(col in new_metadata.columns for col in filename_columns):
            raise ValueError(
                "Metadata does not have a filename column. It should have one of the following columns: 'filename', 'file_name', 'FileName', 'File Name'."
            )

        new_metadata = self._rename_columns(new_metadata, filename_columns)
        metadata = metadata.merge(new_metadata, how="left", on="filename")

        return metadata

    def _open_ifdo_metadata(self):
        metadata = miqtifdo.iFDO_Reader(self.metadata_path).ifdo
        self._validate_ifdo(metadata)
        return metadata

    def _open_csv_metadata(self):
        metadata = pd.read_csv(self.metadata_path)

        if not any(col in metadata.columns for col in index_columns):
            metadata = metadata.reset_index().rename(columns={"index": "ID"})

        metadata = self._rename_columns(metadata, filename_columns, raise_error=True)
        metadata = self._rename_columns(metadata, datetime_columns)
        if "datetime" in metadata.columns:
            metadata["datetime"] = pd.to_datetime(metadata["datetime"])
            metadata.sort_values(by="datetime", inplace=True)

        return metadata

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
        # Return the string representation of the metadata DataFrame
        return repr(self.metadata)

    def _repr_html_(self) -> str:
        message = "This is a instance of 'MetadataParser'<br><br>"

        # Return the HTML representation of the metadata DataFrame for Jupyter
        return message + self.metadata._repr_html_()

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
    #     self.metadata = self.merge_waypoints_to_metadata(waypoints)
    #     return waypoints

    # def merge_waypoints_to_metadata(self, waypoints):
    #     transect = []
    #     metadata = self.metadata.copy()

    #     # Iterate through each photo
    #     for _, photo_row in metadata.iterrows():
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
    #     metadata['transect'] = transect
    #     return metadata
