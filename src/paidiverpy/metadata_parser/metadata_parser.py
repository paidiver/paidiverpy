"""Module for parsing metadata files."""

import logging
import mariqt.sources.ifdo as miqtifdo
import mariqt.tests as miqtt
import pandas as pd
from mariqt.core import IfdoException
from shapely.geometry import Point
from paidiverpy.config.config import Configuration
from paidiverpy.utils import initialise_logging

filename_columns = ["image-filename", "filename", "file_name", "FileName", "File Name"]
index_columns = ["id", "index", "ID", "Index", "Id"]
datetime_columns = ["image-datetime", "datetime", "date_time", "DateTime", "Datetime"]
lat_columns = ["image-latitude",
               "lat",
               "latitude_deg",
               "latitude",
               "Latitude",
               "Latitude_deg",
               "Lat"]
lon_columns = ["image-longitude",
               "lon",
               "longitude_deg",
               "longitude",
               "Longitude",
               "Longitude_deg",
               "Lon"]


class MetadataParser:
    """Class for parsing metadata files.

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
        config: Configuration = None,
        metadata_path: str | None = None,
        metadata_type: str | None = None,
        append_data_to_metadata: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or initialise_logging()
        self.config = config or self._build_config(metadata_path,
                                                   metadata_type,
                                                   append_data_to_metadata)
        self.metadata_type = getattr(self.config.general, "metadata_type", None)
        self.append_data_to_metadata = getattr(self.config.general, "append_data_to_metadata", None)
        self.metadata_path = getattr(self.config.general, "metadata_path", None)
        if not self.metadata_path:
            msg = "Metadata path is not specified."
            raise ValueError(msg)
        if not self.metadata_type:
            msg = "Metadata type is not specified."
            raise ValueError(msg)

        self.metadata = self.open_metadata()
        self.dataset_metadata = None

    def _build_config(self, metadata_path: str, metadata_type: str, append_data_to_metadata: str) -> Configuration:
        """Build a configuration object.

        Args:
            metadata_path (str): Metadata file path.
            metadata_type (str): Metadata file type.
            append_data_to_metadata (str): Additional data file path.

        Returns:
            Configuration: Configuration object.
        """
        general_params = {
            "input_path": "placeholder",
            "output_path": "placeholder",
            "metadata_path": metadata_path,
            "metadata_type": metadata_type,
            "append_data_to_metadata": append_data_to_metadata,
        }
        config = Configuration(input_path="placeholder", output_path="placeholder")
        config.add_config("general", general_params)
        return config

    def open_metadata(self) -> pd.DataFrame:
        """Open metadata file.

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
            msg = "Metadata type is not supported."
            raise ValueError(msg)

        if self.append_data_to_metadata:
            metadata = self._add_data_to_metadata(metadata)

        metadata["flag"] = 0
        return self._process_coordinates(metadata)

    def _process_coordinates(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Process coordinates in the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        metadata = self._rename_columns(metadata, lat_columns)
        metadata = self._rename_columns(metadata, lon_columns)
        if "image-longitude" in metadata.columns and "image-latitude" in metadata.columns:
            metadata["point"] = metadata.apply(lambda x: Point(x["image-longitude"], x["image-latitude"]), axis=1)

        return metadata

    def _rename_columns(self, metadata: pd.DataFrame, columns: list, raise_error: bool = False) -> pd.DataFrame:
        """Rename columns in the metadata.

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
                msg = f"Metadata does not have any column like: {columns}"
                raise ValueError(
                    msg,
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
                metadata = metadata.rename(columns={col: columns[0]})
                columns_1 = columns.copy()
                columns_1.remove(columns_1[0])
                return metadata.drop(columns_1, errors="ignore")
        return None

    def _add_data_to_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Add additional data to the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Raises:
            ValueError: Metadata does not have a filename column.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        new_metadata = pd.read_csv(self.append_data_to_metadata).drop_duplicates(subset="filename", keep="first")

        if not any(col in new_metadata.columns for col in filename_columns):
            msg = f"Metadata does not have a filename column: {filename_columns}"
            raise ValueError(
                msg,
            )

        new_metadata = self._rename_columns(new_metadata, filename_columns)
        return metadata.merge(new_metadata, how="left", on="image-filename")


    def _open_ifdo_metadata(self) -> pd.DataFrame:
        """Open iFDO metadata file.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        metadata_path = self.metadata_path if isinstance(self.metadata_path,
                                                         str) else str(self.metadata_path)
        metadata = miqtifdo.iFDO_Reader(metadata_path).ifdo
        self._validate_ifdo(metadata)
        self.dataset_metadata = metadata["image-set-header"]
        metadata = pd.DataFrame(metadata["image-set-items"]).T.reset_index()
        metadata = metadata.rename(columns={"index": "image-filename"})
        metadata = metadata.reset_index()
        metadata = metadata.rename(columns={"index": "ID"})
        if "image-datetime" in metadata.columns:
            metadata["image-datetime"] = pd.to_datetime(metadata["image-datetime"])
            metadata = metadata.sort_values(by="image-datetime")
        return metadata

    def _open_csv_metadata(self) -> pd.DataFrame:
        """Open CSV metadata file.

        Returns:
            pd.DataFrame: Metadata DataFrame
        """
        metadata = pd.read_csv(self.metadata_path)

        if not any(col in metadata.columns for col in index_columns):
            metadata = metadata.reset_index().rename(columns={"index": "ID"})

        metadata = self._rename_columns(metadata, filename_columns, raise_error=True)
        metadata = self._rename_columns(metadata, datetime_columns)
        if "image-datetime" in metadata.columns:
            metadata["image-datetime"] = pd.to_datetime(metadata["image-datetime"])
            metadata = metadata.sort_values(by="image-datetime")

        return metadata

    @staticmethod
    def _validate_ifdo(ifdo_data: dict) -> None:
        """validate_ifdo method.

        Validates input data against iFDO scheme. Raises an exception if the
        data is invalid.

        Args:
            ifdo_data (Dict): parsed iFDO data.
        """
        miqtt.are_valid_ifdo_fields(ifdo_data["image-set-header"])
        unique_names = miqtt.filesHaveUniqueName(ifdo_data["image-set-items"].keys())
        if not unique_names:
            raise IfdoException({"Validation error": f"Duplicate filenames found: {unique_names}"})

    def __repr__(self) -> str:
        """Return the string representation of the metadata.

        Returns:
            str: String representation of the metadata.
        """
        return repr(self.metadata)

    def _repr_html_(self) -> str:
        """Return the HTML representation of the metadata.

        Returns:
            str: HTML representation of the metadata.
        """
        message = "This is a instance of 'MetadataParser'<br><br>"

        return message + self.metadata._repr_html_()
