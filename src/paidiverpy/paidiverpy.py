"""Main class for the paidiverpy package."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from paidiverpy.config.config import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from utils import DynamicConfig
from utils import get_n_jobs
from utils import initialise_logging


class Paidiverpy:
    """Main class for the paidiverpy package.

    Args:
        config_file_path (str): The path to the configuration file.
        input_path (str): The path to the input files.
        output_path (str): The path to the output files.
        metadata_path (str): The path to the metadata file.
        metadata_type (str): The type of the metadata file.
        metadata (MetadataParser): The metadata object.
        config (Configuration): The configuration object.
        logger (logging.Logger): The logger object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
        track_changes (bool): Whether to track changes. Defaults to True.
        n_jobs (int): The number of n_jobs.
    """

    def __init__(
        self,
        config_file_path: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        metadata_path: str | None = None,
        metadata_type: str | None = None,
        metadata: MetadataParser = None,
        config: Configuration = None,
        logger: logging.Logger | None = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        raise_error: bool = False,
        verbose: int = 2,
        track_changes: bool = True,
        n_jobs: int = 1,
    ):
        if paidiverpy:
            self.logger = paidiverpy.logger
            self.images = paidiverpy.images
            self.config = paidiverpy.config
            self.metadata = paidiverpy.metadata
            self.verbose = paidiverpy.verbose
            self.raise_error = paidiverpy.raise_error
            self.n_jobs = paidiverpy.n_jobs
            self.track_changes = paidiverpy.track_changes
        else:
            self.verbose = verbose
            self.logger = logger or initialise_logging(verbose=self.verbose)
            self.config = config or self._initialize_config(
                config_file_path, input_path, output_path, metadata_path, metadata_type,
            )
            self.metadata = metadata or self._initialize_metadata()
            self.images = images or ImagesLayer(
                output_path=self.config.general.output_path,
            )

            self.raise_error = raise_error
            if not self.config.general.n_jobs:
                self.config.general.n_jobs = n_jobs
            self.n_jobs = get_n_jobs(self.config.general.n_jobs)
            self.track_changes = track_changes

    def _initialize_config(
        self,
        config_file_path: str,
        input_path: str,
        output_path: str,
        metadata_path: str,
        metadata_type: str,
    ) -> Configuration:
        """Initialize the configuration object.

        Args:
            config_file_path (str): Configuration file path.
            input_path (str): input path.
            output_path (str): output path.
            metadata_path (str): metadata path.
            metadata_type (str): metadata type.

        Returns:
            Configuration: The configuration object.
        """
        general_config = {}
        if input_path:
            general_config["input_path"] = input_path
        if output_path:
            general_config["output_path"] = output_path
        if metadata_path:
            general_config["metadata_path"] = metadata_path
        if metadata_type:
            general_config["metadata_type"] = metadata_type

        if config_file_path:
            return Configuration(config_file_path)
        config = Configuration()
        config.add_config("general", general_config)
        return config

    def _initialize_metadata(self) -> MetadataParser:
        """Initialize the metadata object.

        Returns:
            MetadataParser: The metadata object.
        """
        general = self.config.general
        if getattr(general, "metadata_path", None) and getattr(general, "metadata_type", None):
            return MetadataParser(config=self.config, logger=self.logger)
        self.logger.info("Metadata type is not specified. Loading files from the input path.")
        self.logger.info("Metadata will be created from the files in the input path.")
        input_path = Path(general.input_path)
        file_pattern = general.file_name_pattern
        list_of_files = list(input_path.glob(file_pattern))
        metadata = pd.DataFrame(list_of_files, columns=["image-filename"])
        return metadata.reset_index().rename(columns={"index": "ID"})

    def get_metadata(self, flag: int | None = None) -> pd.DataFrame:
        """Get the metadata object.

        Args:
            flag (int, optional): The flag value. Defaults to None.

        Returns:
            pd.DataFrame: The metadata object.
        """
        if isinstance(self.metadata, MetadataParser):
            flag = 0 if flag is None else flag
            if flag == "all":
                if "image-datetime" not in self.metadata.metadata.columns:
                    return self.metadata.metadata.copy()
                return self.metadata.metadata.sort_values("image-datetime").copy()
            if "image-datetime" not in self.metadata.metadata.columns:
                return self.metadata.metadata[self.metadata.metadata["flag"] <= flag].copy()
            return self.metadata.metadata[self.metadata.metadata["flag"] <= flag].sort_values("image-datetime").copy()
        return self.metadata

    def set_metadata(self, metadata: pd.DataFrame) -> None:
        """Set the metadata.

        Args:
            metadata (pd.DataFrame): The metadata object.
        """
        if isinstance(self.metadata, MetadataParser):
            self.metadata.metadata = metadata
        else:
            self.metadata = metadata

    def get_waypoints(self) -> pd.DataFrame:
        """Get the waypoints.

        Raises:
            ValueError: Waypoints are not loaded in the metadata.

        Returns:
            pd.DataFrame: The waypoints
        """
        if isinstance(self.metadata, MetadataParser):
            return self.metadata.waypoints
        msg = "Waypoints are not loaded in the metadata."
        raise ValueError(msg)

    def show_images(self, step_name: str) -> None:
        """Show the images.

        Args:
            step_name (str): The step name.
        """
        for image in self.images[step_name]:
            image.show_image()

    def save_images(
        self,
        step: str | int | None = None,
        by_order: bool = False,
        image_format: str = "png",
    ) -> None:
        """Save the images.

        Args:
            step (Union[str, int], optional): The step name or order. Defaults to None.
            by_order (bool, optional): Whether to save by order. Defaults to False.
            image_format (str, optional): The image format. Defaults to "png".
        """
        last = False
        if step is None:
            last = True
        output_path = self.config.general.output_path
        self.logger.info("Saving images from step: %s", step if not last else "last")
        self.images.save(
            step,
            by_order=by_order,
            last=last,
            output_path=output_path,
            image_format=image_format,
        )
        self.logger.info("Images are saved to: %s", output_path)

    def plot_trimmed_photos(self, new_metadata: pd.DataFrame) -> None:
        """Plot the trimmed photos.

        Args:
            new_metadata (pd.DataFrame): The new metadata.
        """
        metadata = self.get_metadata()
        if "image-longitude" not in metadata.columns or "image-longitude" not in new_metadata.columns:
            self.logger.warning("Longitude and Latitude columns are not found in the metadata.")
            self.logger.warning("Plotting will not be performed.")
            return
        plt.figure(figsize=(20, 10))
        plt.plot(metadata["image-longitude"], metadata["image-latitude"], ".k")
        plt.plot(new_metadata["image-longitude"], new_metadata["image-latitude"], "or")
        plt.legend(["Original", "After Trim"])
        plt.show()

    def clear_steps(self, value: int | str, by_order: bool = True) -> None:
        """Clear steps from the images and metadata.

        Args:
            value (Union[int, str]): Step name or order.
            by_order (bool, optional): Whether to remove by order. Defaults to True.
        """
        if by_order:
            self.images.remove_steps_by_order(value)
        else:
            self.images.remove_steps_by_name(value)
        metadata = self.get_metadata(flag="all")
        metadata.loc[metadata["flag"] >= value, "flag"] = 0
        self.set_metadata(metadata)

    def _calculate_steps_metadata(self, config_part: Configuration) -> dict:
        """Calculate the steps metadata.

        Args:
            config_part (Configuration): The configuration part.

        Returns:
            dict: The steps metadata.
        """
        return dict(config_part.__dict__.items())

    def _get_method_by_mode(self,
                            params: DynamicConfig,
                            method_dict: dict,
                            mode: str) -> tuple:
        """Get the method by mode.

        Args:
            params (DynamicConfig): The parameters.
            method_dict (dict): The method dictionary.
            mode (str): The mode.

        Raises:
            ValueError: Unsupported mode.

        Returns:
            tuple: The method and parameters.
        """
        if mode not in method_dict:
            msg = f"Unsupported mode: {mode}"
            raise ValueError(msg)
        method_info = method_dict[mode]
        if not isinstance(params, method_info["params"]):
            params = method_info["params"](**params)
        method_name = method_info["method"]
        method = getattr(self, method_name)

        return method, params
