"""Module for parsing metadata files."""

import json
import logging
import warnings
from io import BytesIO
from json import JSONDecodeError
from pathlib import Path
import dask.dataframe as dd
import jsonschema
import pandas as pd
from jsonschema import Draft202012Validator
from shapely.geometry import Point
from paidiverpy.config.config import Configuration
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.object_store import define_storage_options
from paidiverpy.utils.object_store import get_file_from_bucket

warnings.filterwarnings("ignore", category=UserWarning)

filename_columns = ["image-filename", "filename", "file_name", "FileName", "File Name"]
index_columns = ["id", "index", "ID", "Index", "Id"]
datetime_columns = ["image-datetime", "datetime", "date_time", "DateTime", "Datetime"]
lat_columns = ["image-latitude", "lat", "latitude_deg", "latitude", "Latitude", "Latitude_deg", "Lat"]
lon_columns = ["image-longitude", "lon", "longitude_deg", "longitude", "Longitude", "Longitude_deg", "Lon"]
depth_columns = ["image-altitude-meters", "depth", "depth_m", "depth_metres", "depth_metre", "depth_meters", "depth_meter"]


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
        self.config = config or self._build_config(metadata_path, metadata_type, append_data_to_metadata)
        self.metadata_type = getattr(self.config.general, "metadata_type", None)
        self.append_data_to_metadata = getattr(self.config.general, "append_data_to_metadata", None)
        self.metadata_path = getattr(self.config.general, "metadata_path", None)
        self.storage_options = define_storage_options(self.metadata_path)

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

    def open_metadata(self) -> dd.DataFrame:
        """Open metadata file.

        Raises:
            ValueError: Metadata type is not supported.

        Returns:
            dd.DataFrame: Metadata DataFrame.
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
        metadata = self._rename_columns(metadata, depth_columns)
        return self._process_coordinates(metadata)

    def _process_coordinates(self, metadata: dd.DataFrame) -> dd.DataFrame:
        """Process coordinates in the metadata.

        Args:
            metadata (dd.DataFrame): Metadata DataFrame.

        Returns:
            dd.DataFrame: Metadata DataFrame.
        """
        metadata = self._rename_columns(metadata, lat_columns)
        metadata = self._rename_columns(metadata, lon_columns)
        if "image-longitude" in metadata.columns and "image-latitude" in metadata.columns:
            metadata["point"] = metadata.apply(lambda x: Point(x["image-longitude"], x["image-latitude"]), axis=1)

        return metadata

    def _rename_columns(self, metadata: dd.DataFrame, columns: list, raise_error: bool = False) -> dd.DataFrame:
        """Rename columns in the metadata.

        Args:
            metadata (dd.DataFrame): Metadata DataFrame.
            columns (list): List of columns to rename.
            raise_error (bool, optional): Raise error if column is not found.
        Defaults to False.

        Raises:
            ValueError: Metadata does not have a column.

        Returns:
            dd.DataFrame: Metadata DataFrame.
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
                return metadata.drop(columns_1, errors="ignore", axis=1)
        return None

    def _add_data_to_metadata(self, metadata: dd.DataFrame) -> dd.DataFrame:
        """Add additional data to the metadata.

        Args:
            metadata (dd.DataFrame): Metadata DataFrame.

        Raises:
            ValueError: Metadata does not have a filename column.

        Returns:
            dd.DataFrame: Metadata DataFrame.
        """
        new_metadata = pd.read_csv(self.append_data_to_metadata).drop_duplicates(subset="filename", keep="first")

        if not any(col in new_metadata.columns for col in filename_columns):
            msg = f"Metadata does not have a filename column: {filename_columns}"
            raise ValueError(
                msg,
            )

        new_metadata = self._rename_columns(new_metadata, filename_columns)
        return metadata.merge(new_metadata, how="left", on="image-filename")

    def _open_ifdo_metadata(self) -> dd.DataFrame:
        """Open iFDO metadata file.

        Returns:
            dd.DataFrame: Metadata DataFrame.
        """
        metadata_path = self.metadata_path if isinstance(self.metadata_path, str) else str(self.metadata_path)

        if self.config.general.is_remote:
            file_bytes = get_file_from_bucket(metadata_path, self.storage_options)
            metadata = json.loads(file_bytes.decode("utf-8"))
        else:
            if is_running_in_docker():
                metadata_filename = Path(metadata_path).name
                metadata_path = f"/app/metadata/{metadata_filename}"
            try:
                with Path(metadata_path).open() as file:
                    metadata = json.load(file)
            except FileNotFoundError as error:
                msg = f"Metadata file not found: {metadata_path}"
                raise FileNotFoundError(msg) from error
            except JSONDecodeError as error:
                msg = f"Metadata file is not a valid JSON file: {metadata_path}. Please check the file"
                self.logger.error("%s: line %s, column %s", msg, error.lineno, error.colno)
                raise JSONDecodeError(msg, doc=error.doc, pos=error.pos) from error
        self._validate_ifdo(metadata)
        self.dataset_metadata = metadata["image-set-header"]
        metadata = dd.from_dict(metadata["image-set-items"], orient="index", npartitions=2)
        metadata = metadata.reset_index()
        metadata = metadata.rename(columns={"index": "image-filename"})
        metadata = metadata.reset_index()
        metadata = metadata.rename(columns={"index": "ID"})

        if "image-datetime" in metadata.columns:
            metadata["image-datetime"] = dd.to_datetime(metadata["image-datetime"])
            metadata = metadata.sort_values(by="image-datetime")
        return metadata.compute()

    def _open_csv_metadata(self) -> dd.DataFrame:
        """Open CSV metadata file.

        Returns:
            dd.DataFrame: Metadata DataFrame
        """
        if self.config.general.is_remote:
            file_bytes = get_file_from_bucket(self.metadata_path, self.storage_options)
            file_bytes = BytesIO(file_bytes)
            df_pandas = pd.read_csv(file_bytes)
            metadata = dd.from_pandas(df_pandas)
        else:
            if is_running_in_docker():
                metadata_filename = Path(self.metadata_path).name
                self.metadata_path = f"/app/metadata/{metadata_filename}"

            metadata = dd.read_csv(self.metadata_path, assume_missing=True)

        if not any(col in metadata.columns for col in index_columns):
            metadata = metadata.reset_index().rename(columns={"index": "ID"})

        metadata = self._rename_columns(metadata, filename_columns, raise_error=True)
        metadata = self._rename_columns(metadata, datetime_columns)
        if "image-datetime" in metadata.columns:
            metadata["image-datetime"] = dd.to_datetime(metadata["image-datetime"])
            metadata = metadata.sort_values(by="image-datetime")

        return metadata.compute()

    def _validate_ifdo(self, ifdo_data: dict) -> None:
        """validate_ifdo method.

        Validates input data against iFDO scheme. Raises an exception if the
        data is invalid.

        Args:
            ifdo_data (Dict): parsed iFDO data.
        """
        ifdo_version = ifdo_data.get("image-set-header", {}).get("image-set-ifdo-version", None)
        if not ifdo_version:
            msg = "No iFDO version found in metadata."
            raise jsonschema.exceptions.ValidationError(msg)
        schema_file_path = f"https://www.marine-imaging.com/fair/schemas/ifdo-{ifdo_version}.json"
        schema = json.loads(get_file_from_bucket(schema_file_path))
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(ifdo_data), key=lambda e: e.path)
        if errors:
            msg_warn = "Failed to validate the IFDO metadata.\n"
            msg_warn += "You can continue, but some functions may not work properly.\n"
            msg_warn += "Please set verbose to 3 (DEBUG) to see the validation errors."
            self.logger.warning(msg_warn)
            msg_debug = "Validation errors with the metadata:\n"
            for error in errors:
                msg_debug += f"{MetadataParser.format_error(error.path)}: {error.message}\n"
            self.logger.debug(msg_debug)
        else:
            self.logger.info("Metadata file is valid.")

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
        metadata = self.metadata

        return message + metadata._repr_html_()

    @staticmethod
    def format_error(text: list) -> str:
        """Format error message.

        Args:
            text (list): List of error messages.

        Returns:
            str: Formatted error message.
        """
        if len(text) > 3:  # noqa: PLR2004
            return f"...{'.'.join(map(str, text[-3:]))}"
        return ".".join(map(str, text))
