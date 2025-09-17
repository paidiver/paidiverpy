"""Module for parsing metadata files."""

import json
import logging
import uuid
import warnings
from io import BytesIO
from json import JSONDecodeError
from pathlib import Path
from typing import Any
import dask.dataframe as dd
import pandas as pd
from shapely.geometry import Point
from paidiverpy.config.configuration import Configuration
from paidiverpy.metadata_parser.ifdo_tools import convert_to_ifdo
from paidiverpy.metadata_parser.ifdo_tools import format_ifdo_validation_error
from paidiverpy.metadata_parser.ifdo_tools import validate_ifdo
from paidiverpy.utils import formating_html
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.object_store import define_storage_options
from paidiverpy.utils.object_store import get_file_from_bucket
from paidiverpy.utils.object_store import path_is_remote

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("paidiverpy")


class MetadataParser:
    """Class for parsing metadata files.

    Args:
        config (Configuration | None): Configuration object.
        use_dask (bool): Whether to use Dask for parallel processing.
        metadata_path (str): Path to the metadata file.
        metadata_type (str): Type of the metadata file.
        append_data_to_metadata (str): Path to the file with additional data.

    Raises:
        ValueError: Metadata path is not specified.
        ValueError: Metadata type is not specified.
    """

    def __init__(
        self,
        config: Configuration | None = None,
        use_dask: bool = False,
        metadata_path: str | None = None,
        metadata_type: str | None = None,
        metadata_conventions: str | None = None,
        append_data_to_metadata: str | None = None,
    ):
        self.config = config or self._build_config(metadata_path, metadata_type, metadata_conventions, append_data_to_metadata)
        self.metadata_type = getattr(self.config.general, "metadata_type", None)
        self.append_data_to_metadata = getattr(self.config.general, "append_data_to_metadata", None)
        self.metadata_path = getattr(self.config.general, "metadata_path", None)
        self.dataset_metadata: dict[str, Any] = {}
        self.use_dask = use_dask

        if self.metadata_path is None and not self.append_data_to_metadata:
            self.metadata = self._load_from_file_list()
        else:
            self.storage_options = define_storage_options(self.metadata_path)
            self.metadata_conventions = self._calculate_metadata_conventions()
            self.metadata = self.open_metadata()

    def _load_from_file_list(self) -> pd.DataFrame:
        """Load metadata from a list of files.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        logger.info(
            "Metadata type is not specified. Metadata will be created from the files in the input path.",
        )
        input_path = Path(self.config.general.input_path)
        file_pattern = self.config.general.file_name_pattern
        list_of_files = list(input_path.glob(file_pattern))
        metadata = pd.DataFrame(list_of_files, columns=["filename"])
        metadata["ID"] = pd.Series((str(uuid.uuid4()) for _ in range(len(metadata))), index=metadata.index)
        return self._prepare_metadata(metadata)

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
                logger.warning("Metadata conventions file not found: %s", metadata_conventions)
                logger.warning("Using default metadata conventions file: %s", file_path)
        with file_path.open() as file:
            return json.load(file)

    def _build_config(
        self, metadata_path: str | None, metadata_type: str | None, metadata_conventions: str | None, append_data_to_metadata: str | None
    ) -> Configuration:
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

    def open_metadata(self) -> pd.DataFrame:
        """Open metadata file.

        Raises:
            ValueError: Metadata type is not supported.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        metadata = self._open_ifdo_metadata() if self.metadata_type == "IFDO" else self._open_csv_metadata()
        if self.append_data_to_metadata:
            metadata = self._add_data_to_metadata(metadata)
        if "image-set-uuid" not in self.dataset_metadata:
            self.dataset_metadata["image-set-uuid"] = str(uuid.uuid4())
            logger.info("No dataset UUID found in the dataset metadata. A new UUID has been generated: %s", self.dataset_metadata["image-set-uuid"])
        metadata["flag"] = 0
        return self._prepare_metadata(metadata)

    def set_metadata(self, metadata: pd.DataFrame | None = None, dataset_metadata: dict[str, Any] | None = None) -> None:
        """Set the metadata.

        Args:
            metadata (pd.DataFrame | None): The metadata to set.
            dataset_metadata (dict | None): The dataset metadata to set.
        """
        if metadata is not None:
            self.metadata = metadata
        if dataset_metadata is not None:
            self.dataset_metadata.update(dataset_metadata)

    def export_metadata(
        self,
        output_format: str = "csv",
        output_path: str = "metadata",
        metadata: pd.DataFrame | None = None,
        dataset_metadata: dict[str, Any] | None = None,
        from_step: int = -1,
    ) -> None:
        """Export metadata to a file.

        Args:
            output_format (str, optional): Format of the output file. It can be
        "csv", "json", "IFDO", or "croissant". Defaults to "csv".
            output_path (str, optional): Path to the output file. Defaults to "metadata".
            metadata (pd.DataFrame, optional): Metadata DataFrame. Defaults to None.
            dataset_metadata (dict, optional): Dataset metadata. Defaults to None.
            from_step (int, optional): Step from which to export metadata. Defaults to None, which means last step.
        """
        if metadata is None:
            metadata = self.metadata
        if dataset_metadata is None:
            dataset_metadata = self.dataset_metadata
        if output_format.lower() not in ["csv", "json", "ifdo", "croissant"]:
            logger.error("Unsupported output format: %s", output_format)
            raise_value_error(f"Unsupported output format: {output_format}")
        try:
            MetadataParser.convert_metadata_to(
                dataset_metadata=dataset_metadata, metadata=metadata, output_path=output_path, output_format=output_format, from_step=from_step
            )
            logger.info("Metadata exported to %s file in format %s", output_path, output_format)
        except Exception as error:  # noqa: BLE001
            logger.error("Failed to export metadata: %s", error)
            raise_value_error(f"Failed to export metadata: {error}")

    def _prepare_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Prepare metadata for processing.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        errors: list[str] = []
        metadata = self._rename_columns(metadata, "image-altitude-meters", errors=errors)
        metadata = self._rename_columns(metadata, "image-depth", errors=errors)
        metadata = self._rename_columns(metadata, "image-latitude", errors=errors)
        metadata = self._rename_columns(metadata, "image-longitude", errors=errors)
        metadata = self._rename_columns(metadata, "image-camera-pitch-degrees", errors=errors)
        metadata = self._rename_columns(metadata, "image-camera-roll-degrees", errors=errors)
        for error in errors:
            logger.warning(error)
        if errors:
            logger.warning("Some functions may not work properly.")
        if "image-longitude" in metadata.columns and "image-latitude" in metadata.columns:
            metadata["point"] = metadata.apply(lambda x: Point(x["image-longitude"], x["image-latitude"]), axis=1)
        return metadata

    def _rename_columns(self, metadata: pd.DataFrame, column_name: str, errors: list[str] | None = None, raise_error: bool = False) -> pd.DataFrame:
        """Rename columns in the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.
            column_name (str): Column name to rename.
            columns (list): List of columns to rename.
            errors (list, optional): List of errors to append to.
            raise_error (bool, optional): Raise error if column is not found.
        Defaults to False.

        Raises:
            ValueError: Metadata does not have a column.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        if column_name not in self.metadata_conventions:
            raise_value_error(f"Column {column_name} is not in the metadata conventions file. Please add to the file and try again.")
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
            logger.warning(msg)
        return metadata

    def _add_data_to_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Add additional data to the metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Raises:
            ValueError: Metadata does not have a filename column.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        # new_metadata = dd.read_csv(self.append_data_to_metadata) if self.use_dask else pd.read_csv(self.append_data_to_metadata)
        new_metadata = pd.read_csv(self.append_data_to_metadata)
        try:
            new_metadata = self._rename_columns(new_metadata, "filename", raise_error=True)
        except ValueError:
            logger.warning("The new metadata will not be added to the metadata.")
            return metadata
        new_metadata = new_metadata.drop_duplicates(subset="filename", keep="first")
        return metadata.merge(new_metadata, how="left", on="filename")

    def _open_ifdo_metadata(self) -> pd.DataFrame:
        """Open iFDO metadata file.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        metadata_path = self.metadata_path if isinstance(self.metadata_path, str) else str(self.metadata_path)

        if path_is_remote(metadata_path):
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
                logger.error("%s: line %s, column %s", msg, error.lineno, error.colno)
                raise JSONDecodeError(msg, doc=error.doc, pos=error.pos) from error
        self._validate_ifdo(metadata)
        self.dataset_metadata = metadata["image-set-header"]
        # if self.use_dask:
        #     metadata = dd.from_dict(metadata["image-set-items"], orient="index", npartitions=2)
        # else:
        #     metadata = pd.DataFrame.from_dict(metadata["image-set-items"], orient="index")
        metadata = pd.DataFrame.from_dict(metadata["image-set-items"], orient="index")
        metadata = metadata.reset_index().rename(columns={"index": "filename"})
        if "image-uuid" in metadata.columns:
            metadata = metadata.rename(columns={"image-uuid": "ID"})
        else:
            metadata["ID"] = pd.Series((str(uuid.uuid4()) for _ in range(len(metadata))), index=metadata.index)

        return self._handle_datetime(metadata)

    def _open_csv_metadata(self) -> pd.DataFrame:
        """Open CSV metadata file.

        Returns:
            pd.DataFrame: Metadata DataFrame
        """
        if path_is_remote(self.metadata_path):
            file_bytes = get_file_from_bucket(self.metadata_path, self.storage_options)
            file_bytes = BytesIO(file_bytes)
            df_pandas = pd.read_csv(file_bytes)
            metadata = df_pandas
            # metadata = dd.from_pandas(df_pandas, npartitions=2) if self.use_dask else df_pandas
        else:
            if is_running_in_docker() and not self.config.general.sample_data:
                metadata_filename = Path(self.metadata_path).name
                self.metadata_path = f"/app/metadata/{metadata_filename}"

            # metadata = dd.read_csv(self.metadata_path, assume_missing=True) if self.use_dask else pd.read_csv(self.metadata_path)
            metadata = pd.read_csv(self.metadata_path)

        metadata = self._rename_columns(metadata, "filename", raise_error=True)
        try:
            metadata = self._rename_columns(metadata, "ID", raise_error=True)
        except ValueError:
            metadata["ID"] = pd.Series((str(uuid.uuid4()) for _ in range(len(metadata))), index=metadata.index)
        metadata = self._rename_columns(metadata, "image-datetime")

        return self._handle_datetime(metadata)

    def _handle_datetime(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Handle datetime conversion and sorting.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.

        Returns:
            pd.DataFrame: Updated metadata DataFrame.
        """
        if "image-datetime" not in metadata.columns:
            logger.warning("Metadata does not have a datetime column")
            logger.warning("Some functions may not work properly.")
        else:
            # if self.use_dask:
            #     metadata["image-datetime"] = dd.to_datetime(metadata["image-datetime"])
            # else:
            #     metadata["image-datetime"] = pd.to_datetime(metadata["image-datetime"])
            metadata["image-datetime"] = pd.to_datetime(metadata["image-datetime"])
            metadata = metadata.sort_values(by="image-datetime")
        return metadata

    def _validate_ifdo(self, metadata: dict[str, Any]) -> None:
        """Validate iFDO metadata.

        Args:
            metadata (dict): Metadata dictionary.
        """
        errors = validate_ifdo(ifdo_data=metadata)
        if errors:
            msg_warn = "Failed to validate the IFDO metadata.\n"
            msg_warn += "You can continue, but some functions may not work properly.\n"
            msg_warn += "Please set verbose to 3 (DEBUG) to see the validation errors."
            logger.warning(msg_warn)
            msg_debug = "Validation errors with the metadata:\n"
            for error in errors:
                msg_debug += f"{format_ifdo_validation_error(error['path'])}: {error['message']}\n"
            logger.debug(msg_debug)
        else:
            logger.info("Metadata file is valid.")

    def compute(self) -> None:
        """Compute the metadata if it is a Dask DataFrame."""
        if isinstance(self.metadata, dd.DataFrame):
            self.metadata = self.metadata.compute()

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
        return formating_html.metadata_repr(self)

    @staticmethod
    def convert_metadata_to(
        dataset_metadata: dict[str, Any], metadata: pd.DataFrame, output_path: str, output_format: str, from_step: int = -1
    ) -> None:
        """Convert metadata to specified format.

        Args:
            dataset_metadata (dict): Dataset metadata.
            metadata (pd.DataFrame): Metadata to convert.
            output_path (str): Path to save the converted metadata.
            output_format (str): Type of metadata to convert to. It can be "csv",
        "json", "IFDO", or "croissant".
            from_step (int): Step to filter metadata. Default is -1, which means the last step.
        """
        if "flag" in metadata.columns:
            metadata = metadata[metadata["flag"] == 0] if from_step < 0 else metadata[(metadata["flag"] == 0) | (metadata["flag"] > from_step)]
            metadata = metadata.drop(columns=["flag"], errors="ignore")
        if "point" in metadata.columns:
            metadata = metadata.drop(columns=["point"], errors="ignore")
        empty_cols = metadata.columns[metadata.isna().all()]
        metadata = metadata.drop(columns=empty_cols)
        if output_format.lower() in ["csv", "json"]:
            output_path = f"{output_path}.{output_format}"
            metadata = MetadataParser.group_metadata_and_dataset_metadata(metadata, dataset_metadata)
            metadata.to_csv(output_path, index=False) if output_format == "csv" else metadata.to_json(output_path, orient="records")
        elif output_format.lower() == "ifdo":
            output_path = f"{output_path}.json"
            convert_to_ifdo(dataset_metadata, metadata, output_path)
        elif output_format.lower() == "croissant":
            # output_path = f"{output_path}.json"
            # convert_to_croissant(dataset_metadata, metadata, output_path)
            msg = "Croissant format is not implemented yet."
            raise NotImplementedError(msg)
        else:
            raise_value_error(f"Unsupported output format: {output_format}")

    @staticmethod
    def group_metadata_and_dataset_metadata(
        metadata: pd.DataFrame,
        dataset_metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Group metadata and dataset metadata.

        Args:
            metadata (pd.DataFrame): Metadata DataFrame.
            dataset_metadata (dict): Dataset metadata.
            metadata_type (str): Metadata type. Defaults to "IFDO".

        Returns:
            pd.DataFrame: Combined metadata DataFrame.
        """
        # metadata = metadata.compute() if isinstance(metadata, dd.DataFrame) else metadata
        for key, value in dataset_metadata.items():
            if key not in metadata.columns:
                metadata[key] = value
        for col in metadata.select_dtypes(include=["object"]).columns:
            metadata[col] = metadata[col].astype(str)
        return metadata

    # @staticmethod
    # def metadata_to_exif(
    #     filename: str,
    #     metadata: pd.DataFrame,
    #     image_format: str = "png",
    # ) -> None | dict[str, Any]:
    #     """Convert metadata to EXIF format.

    #     Args:
    #         filename (str): Filename to convert.
    #         metadata (pd.DataFrame): Metadata DataFrame.
    #         image_format (str): Image format. Defaults to "png".

    #     Returns:
    #         None | dict: EXIF data or None if not found.
    #     """
    #     _ = filename
    #     exif_dict = None
    #     if metadata is not None:
    #         if image_format.lower() == "jpeg":
    #             pass
    #             # exif_dict = {"0th": {piexif.ImageIFD.Artist: "Tobias"}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    #             # exif_dict = piexif.dump(exif_dict)
    #             # exif_dict = piexif.load(filename)
    #         elif image_format.lower() == "tiff":
    #             # For TIFF/PNG, we can use PIL to extract EXIF-like data
    #             # Note: This is very limited and may not cover all EXIF tags
    #             # TIFF/PNG does not have a standard EXIF format
    #             # This is a placeholder for actual implementation
    #             exif_dict = TiffImagePlugin.ImageFileDirectory_v2()
    #             for key, value in metadata.items():
    #                 if isinstance(value, str | int | float):
    #                     exif_dict[key] = value
    #         elif image_format.lower() == "png":
    #             # PNG does not have EXIF, but we can use tEXt chunks
    #             exif_dict = {}
    #     return exif_dict

    # @staticmethod
    # def df2dataarray(metadata: pd.DataFrame) -> xr.DataArray:
    #     """Convert metadata DataFrame to xarray DataArray.

    #     Args:
    #         metadata (pd.DataFrame): The metadata DataFrame.

    #     Returns:
    #         xr.DataArray: The metadata xarray DataArray.
    #     """
    #     filenames = metadata["filename"].to_numpy()
    #     flags = metadata["flag"].to_numpy()
    #     return xr.DataArray(
    #         metadata.to_dict(orient="records"),
    #         dims=["filename"],
    #         coords={"filename": filenames, "flag": (["filename"], flags)},
    #     )
