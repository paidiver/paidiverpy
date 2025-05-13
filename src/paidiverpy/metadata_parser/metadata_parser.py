"""Module for parsing metadata files."""

import json
import logging
import warnings
from io import BytesIO
from json import JSONDecodeError
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from shapely.geometry import Point
from paidiverpy.config.config import Configuration
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.object_store import define_storage_options
from paidiverpy.utils.object_store import get_file_from_bucket

warnings.filterwarnings("ignore", category=UserWarning)


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
        metadata_conventions: str | None = None,
        append_data_to_metadata: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or initialise_logging()
        self.config = config or self._build_config(metadata_path, metadata_type, metadata_conventions, append_data_to_metadata)
        self.metadata_type = getattr(self.config.general, "metadata_type", None)
        self.append_data_to_metadata = getattr(self.config.general, "append_data_to_metadata", None)
        self.metadata_path = getattr(self.config.general, "metadata_path", None)
        self.storage_options = define_storage_options(self.metadata_path)
        self.metadata_conventions = self._calculate_metadata_conventions()

        self.metadata = self.open_metadata()
        self.dataset_metadata = None
        self._define_additional_attributes()

    def _define_additional_attributes(self) -> None:
        """Set new attributes for the metadata parser."""
        self.trimmed_polygon = None

    def set_new_attributes(self, **kwargs: dict) -> None:
        """Set new attributes for the metadata parser."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _calculate_metadata_conventions(self) -> str:
        """Calculate metadata conventions.

        Returns:
            str: Metadata conventions.
        """
        file_path = Path(__file__).parent / "metadata_conventions.json"
        metadata_conventions = getattr(self.config.general, "metadata_conventions", None)
        if metadata_conventions:
            metadata_conventions = metadata_conventions if isinstance(metadata_conventions, Path) else Path(metadata_conventions)
            if metadata_conventions.is_file():
                file_path = metadata_conventions
            else:
                self.logger.warning("Metadata conventions file not found: %s", metadata_conventions)
                self.logger.warning("Using default metadata conventions file: %s", file_path)
        with file_path.open() as file:
            return json.load(file)

    def _build_config(self, metadata_path: str, metadata_type: str, metadata_conventions: str, append_data_to_metadata: str) -> Configuration:
        """Build a configuration object.

        Args:
            metadata_path (str): Metadata file path.
            metadata_type (str): Metadata file type.
            metadata_conventions (str): Metadata conventions.
            append_data_to_metadata (str): Additional data file path.

        Returns:
            Configuration: Configuration object.
        """
        general_params = {
            "input_path": "placeholder",
            "output_path": "placeholder",
            "metadata_path": metadata_path,
            "metadata_type": metadata_type,
            "metadata_conventions": metadata_conventions,
            "append_data_to_metadata": append_data_to_metadata,
        }
        return Configuration(add_general=general_params)

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
        if self.append_data_to_metadata:
            metadata = self._add_data_to_metadata(metadata)

        metadata["flag"] = 0
        return self._prepare_metadata(metadata)

    def _prepare_metadata(self, metadata: dd.DataFrame) -> dd.DataFrame:
        """Prepare metadata for processing.

        Args:
            metadata (dd.DataFrame): Metadata DataFrame.

        Returns:
            dd.DataFrame: Metadata DataFrame.
        """
        errors = []
        metadata = self._rename_columns(metadata, "image-altitude-meters", errors=errors)
        metadata = self._rename_columns(metadata, "image-latitude", errors=errors)
        metadata = self._rename_columns(metadata, "image-longitude", errors=errors)
        metadata = self._rename_columns(metadata, "image-camera-pitch-degrees", errors=errors)
        metadata = self._rename_columns(metadata, "image-camera-roll-degrees", errors=errors)
        for error in errors:
            self.logger.warning(error)
        if errors:
            self.logger.warning("Some functions may not work properly.")
        if "image-longitude" in metadata.columns and "image-latitude" in metadata.columns:
            metadata["point"] = metadata.apply(lambda x: Point(x["image-longitude"], x["image-latitude"]), axis=1)
        return metadata

    def _rename_columns(self, metadata: dd.DataFrame, column_name: str, errors: list | None = None, raise_error: bool = False) -> dd.DataFrame:
        """Rename columns in the metadata.

        Args:
            metadata (dd.DataFrame): Metadata DataFrame.
            column_name (str): Column name to rename.
            columns (list): List of columns to rename.
            errors (list, optional): List of errors to append to.
            raise_error (bool, optional): Raise error if column is not found.
        Defaults to False.

        Raises:
            ValueError: Metadata does not have a column.

        Returns:
            dd.DataFrame: Metadata DataFrame.
        """
        columns = self.metadata_conventions[column_name].copy()
        columns.append(column_name)
        for col in columns:
            if col in metadata.columns:
                metadata = metadata.rename(columns={col: columns[-1]})
                columns_1 = columns.copy()
                columns_1.remove(columns_1[-1])
                return metadata.drop(columns_1, errors="ignore", axis=1)
        if column_name == "ID":
            msg = f"Metadata does not have a {columns[0]} type column. This column will be created. \n"
        else:
            msg = f"Metadata does not have a {columns[0]} type column. It should have one of the following columns: {columns}. \n"
        if raise_error:
            raise ValueError(
                msg,
            )
        if errors is not None:
            errors.append(msg)
        else:
            self.logger.warning(msg)
        return metadata

    def _add_data_to_metadata(self, metadata: dd.DataFrame) -> dd.DataFrame:
        """Add additional data to the metadata.

        Args:
            metadata (dd.DataFrame): Metadata DataFrame.

        Raises:
            ValueError: Metadata does not have a filename column.

        Returns:
            dd.DataFrame: Metadata DataFrame.
        """
        new_metadata = pd.read_csv(self.append_data_to_metadata)
        try:
            new_metadata = self._rename_columns(new_metadata, "image-filename", raise_error=True)
        except ValueError:
            self.logger.warning("The new metadata will not be added to the metadata.")
            return metadata
        new_metadata = new_metadata.drop_duplicates(subset="image-filename", keep="first")
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

        if "image-datetime" not in metadata.columns:
            self.logger.warning("Metadata does not have a datetime column")
            self.logger.warning("Some functions may not work properly.")

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
        metadata = self._rename_columns(metadata, "image-filename", raise_error=True)
        try:
            metadata = self._rename_columns(metadata, "ID", raise_error=True)
        except ValueError:
            metadata = metadata.reset_index().rename(columns={"index": "ID"})
        metadata = self._rename_columns(metadata, "image-datetime")
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
            raise ValidationError(msg)
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
