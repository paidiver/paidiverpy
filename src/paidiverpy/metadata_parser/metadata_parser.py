""" Module for parsing metadata files. """
import logging
from typing import Dict
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
    """ Class for parsing metadata files.
    
    Args:
        config (Configuration): Configuration object.
        metadata_path (str): Path to the metadata file.
        metadata_type (str): Type of the metadata file.
        append_data_to_metadata (str): Path to the file with additional data.
        logger (logging.Logger): Logger object.
    
    Raises:
        ValueError: Metadata path is not specified.
        ValueError: Metadata type is not specified.
    """
    def __init__(
        self,
        config: Configuration=None,
        metadata_path: str=None,
        metadata_type: str=None,
        append_data_to_metadata: str=None,
        logger: logging.Logger=None,
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

    def _build_config(self,
                      metadata_path: str,
                      metadata_type: str,
                      append_data_to_metadata: str) -> Configuration:
        """ Build a configuration object.

        Args:
            metadata_path (str): Metadata file path.
            metadata_type (str): Metadata file type.
            append_data_to_metadata (str): Additional data file path.

        Returns:
            Configuration: Configuration object.
        """
        general_params = {
            "metadata_path": metadata_path,
            "metadata_type": metadata_type,
            "append_data_to_metadata": append_data_to_metadata,
        }
        config = Configuration()
        config.add_config("general", general_params)
        return config

    def open_metadata(self) -> pd.DataFrame:
        """ Open metadata file.

        Raises:
            ValueError: Metadata type is not supported.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
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

    def _process_coordinates(self,
                             metadata: pd.DataFrame) -> pd.DataFrame:
        """ Process coordinates in the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """

        metadata = self._rename_columns(metadata, lat_columns)
        metadata = self._rename_columns(metadata, lon_columns)
        if "lon" in metadata.columns and "lat" in metadata.columns:
            metadata["point"] = metadata.apply(
                lambda x: Point(x["lon"], x["lat"]), axis=1
            )

        return metadata

    def _rename_columns(self,
                        metadata: pd.DataFrame,
                        columns: list,
                        raise_error: bool=False) -> pd.DataFrame:
        """ Rename columns in the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.
            columns (list): List of columns to rename.
            raise_error (bool, optional): Raise error if column is not found.
        Defaults to False.

        Raises:
            ValueError: Metadata does not have a column.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
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

    def _add_data_to_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """ Add additional data to the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Raises:
            ValueError: Metadata does not have a filename column.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
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

    def _open_ifdo_metadata(self) -> pd.DataFrame:
        """ Open iFDO metadata file.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        metadata = miqtifdo.iFDO_Reader(self.metadata_path).ifdo
        self._validate_ifdo(metadata)
        metadata = pd.DataFrame(metadata["image-set-items"]).T
        return metadata

    def _open_csv_metadata(self) -> pd.DataFrame:
        """ Open CSV metadata file.
        
        Returns:
            pd.DataFrame: Metadata DataFrame
        """
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
        """Validates input data against iFDO scheme. Raises an exception if the
        data is invalid.

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
        """ Return the string representation of the metadata.

        Returns:
            str: String representation of the metadata.
        """
        return repr(self.metadata)

    def _repr_html_(self) -> str:
        """ Return the HTML representation of the metadata.

        Returns:
            str: HTML representation of the metadata.
        """
        message = "This is a instance of 'MetadataParser'<br><br>"

        return message + self.metadata._repr_html_()
