"""Main class for the paidiverpy package."""

import logging
from functools import partial
from pathlib import Path
import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from distributed import LocalCluster
from tqdm import tqdm
from paidiverpy.config.config import Configuration
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils.dynamic_classes import DynamicConfig
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.parallellisation import get_client
from paidiverpy.utils.parallellisation import get_n_jobs


class Paidiverpy:
    """Main class for the paidiverpy package.

    Args:
        config_params (dict | ConfigParams, optional): The configuration parameters.
            It can contain the following keys / attributes:
            - input_path (str): The path to the input files.
            - output_path (str): The path to the output files.
            - metadata_path (str): The path to the metadata file.
            - metadata_type (str): The type of the metadata file.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of n_jobs.
        config_file_path (str, optional): The path to the configuration file.
        config (Configuration, optional): The configuration object.
        metadata (MetadataParser, optional): The metadata object.
        images (ImagesLayer, optional): The images object.
        client (Client, optional): The Dask client object.
        paidiverpy (Paidiverpy, optional): The paidiverpy object.
        track_changes (bool): Whether to track changes. Defaults to None, which means
            it will be set to the value of the configuration file.
        logger (logging.Logger, optional): The logger object.
        raise_error (bool, optional): Whether to raise an error.
        verbose (int, optional): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        client: Client | None = None,
        paidiverpy: "Paidiverpy" = None,
        track_changes: bool | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        if paidiverpy:
            self._set_variables_from_paidiverpy(paidiverpy)
        else:
            self.raise_error = raise_error
            self.verbose = verbose
            self.logger = logger or initialise_logging(verbose=self.verbose)
            try:
                self.config = config or self._initialise_config(config_file_path, config_params)
            except Exception as error:
                msg = f"{error}"
                self.logger.error(msg)
                raise
            self.metadata = metadata or self._initialize_metadata()
            self.images = images or ImagesLayer(
                output_path=self.config.general.output_path,
            )
            if not client:
                result = get_client(self.config.general.client, self.config.general.n_jobs)
                if isinstance(result, tuple):
                    self.client, self.job_id = result
                else:
                    self.client = result
                    self.job_id = None
            else:
                self.client = client
                self.job_id = None
            self.n_jobs = get_n_jobs(self.config.general.n_jobs)
            self.track_changes = self.config.general.track_changes
        if track_changes is not None:
            self.track_changes = track_changes
        self.layer_methods = None

    def run(self, add_new_step: bool = True) -> ImagesLayer | None:
        """Run the paidiverpy pipeline.

        Args:
            add_new_step (bool, optional): Whether to add a new step. Defaults to True.

        Returns:
            ImagesLayer | None: The images object.
        """
        mode = self.step_metadata.get("mode")
        if not mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, self.layer_methods, mode)
        images = self.images.get_step(step=len(self.images.images) - 1, by_order=True)
        image_list = self.process_sequentially(images, method, params) if self.n_jobs == 1 else self.process_parallel(images, method, params)
        if not test:
            self.step_name = f"color_{self.config_index}" if not self.step_name else self.step_name
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=image_list,
                    step_metadata=self.step_metadata,
                    metadata=self.get_metadata(),
                    track_changes=self.track_changes,
                )
                return None
            self.images.images[-1] = image_list
            return self.images
        return None

    def process_sequentially(self, images: list[np.ndarray], method: callable, params: dict, custom: bool = False) -> list[np.ndarray]:
        """Process the images sequentially.

        Method to process the images sequentially.

        Args:
            images (List[np.ndarray]): The list of images to process.
            method (callable): The method to apply to the images.
            params (dict): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            List[np.ndarray]: The list of processed images.
        """
        func = partial(method, params=params)
        return [func(img).process() if custom else func(img) for img in tqdm(images, desc="Processing images")]

    def process_parallel(
        self,
        images: list[da.core.Array],
        method: callable,
        params: DynamicConfig,
        custom: bool = False,
    ) -> list[np.ndarray]:
        """Process the images in parallel.

        Method to process the images in parallel.

        Args:
            images (List[da.core.Array]): The list of images to process.
            method (callable): The method to apply to the images.
            params (DynamicConfig): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            List[da.core.Array]: The list of processed images.
        """
        func = partial(method, params=params)
        if self.client:
            if isinstance(self.client.cluster, LocalCluster):
                delayed_images = [dask.delayed(func)(img) for img in images]
                futures = self.client.compute(delayed_images)
            else:
                futures = [self.client.submit(func, img) for img in images]

            with ProgressBar():
                results = self.client.gather(futures)
            return [da.from_array(img.process() if custom else img) for img in results]

        delayed_images = [dask.delayed(func)(img) for img in images]

        with dask.config.set(scheduler="threads", num_workers=self.n_jobs), ProgressBar():
            results = dask.compute(*delayed_images)

        return [da.from_array(img.process() if custom else img) for img in results]

    def _set_variables_from_paidiverpy(self, paidiverpy: "Paidiverpy") -> None:
        """Set the variables from the paidiverpy object.

        Args:
            paidiverpy (Paidiverpy): The paidiverpy object.
        """
        self.logger = paidiverpy.logger
        self.images = paidiverpy.images
        self.config = paidiverpy.config
        self.metadata = paidiverpy.metadata
        self.verbose = paidiverpy.verbose
        self.raise_error = paidiverpy.raise_error
        self.n_jobs = paidiverpy.n_jobs
        self.track_changes = paidiverpy.track_changes
        self.client = paidiverpy.client
        self.job_id = paidiverpy.job_id

    def _initialise_config(
        self,
        config_file_path: str,
        config_params: ConfigParams | dict,
    ) -> Configuration:
        """Initialize the configuration object.

        Args:
            config_file_path (str): Configuration file path.
            config_params (ConfigParams | dict): Configuration parameters.

        Returns:
            Configuration: The configuration object.
        """
        if config_file_path:
            return Configuration(config_file_path)
        general_config = {}
        config_params = ConfigParams(config_params) if isinstance(config_params, dict) else config_params
        if config_params.input_path:
            general_config["input_path"] = config_params.input_path
        if config_params.output_path:
            general_config["output_path"] = config_params.output_path
        if config_params.metadata_path:
            general_config["metadata_path"] = config_params.metadata_path
        if config_params.metadata_type:
            general_config["metadata_type"] = config_params.metadata_type
        if config_params.track_changes:
            general_config["track_changes"] = config_params.track_changes
        if config_params.n_jobs:
            general_config["n_jobs"] = config_params.n_jobs
        config = Configuration(logger=self.logger)
        config.add_config("general", general_config)
        return config

    def _initialize_metadata(self) -> MetadataParser:
        """Initialize the metadata object.

        Returns:
            MetadataParser: The metadata object.
        """
        general = self.config.general
        if getattr(general, "metadata_path", None) and getattr(
            general,
            "metadata_type",
            None,
        ):
            return MetadataParser(config=self.config, logger=self.logger)
        self.logger.info(
            "Metadata type is not specified. Loading files from the input path.",
        )
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
            step (str | int, optional): The step name or order. Defaults to None.
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
            client=self.client,
            n_jobs=self.n_jobs,
            logger=self.logger,
        )
        self.logger.info("Images are saved to: %s", output_path)

    def remove_images(self) -> None:
        """Remove output images from the output path."""
        output_path = self.config.general.output_path
        self.logger.info("Removing images from the output path: %s", output_path)
        self.images.remove(output_path)

    def plot_trimmed_photos(self, new_metadata: pd.DataFrame) -> None:
        """Plot the trimmed photos.

        Args:
            new_metadata (pd.DataFrame): The new metadata.
        """
        metadata = self.get_metadata()
        if "image-longitude" not in metadata.columns or "image-longitude" not in new_metadata.columns:
            self.logger.warning(
                "Longitude and Latitude columns are not found in the metadata.",
            )
            self.logger.warning("Plotting will not be performed.")
            return
        plt.figure(figsize=(20, 10))
        plt.plot(metadata["image-longitude"], metadata["image-latitude"], ".k")
        plt.plot(new_metadata["image-longitude"], new_metadata["image-latitude"], "or")
        plt.legend(["Original", "After Trim"])
        plt.show(block=False)

    def clear_steps(self, value: int | str, by_order: bool = True) -> None:
        """Clear steps from the images and metadata.

        Args:
            value (int | str): Step name or order.
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

    def _get_method_by_mode(
        self,
        params: DynamicConfig,
        method_dict: dict,
        mode: str,
        class_method: bool = True,
    ) -> tuple:
        """Get the method by mode.

        Args:
            params (DynamicConfig): The parameters.
            method_dict (dict): The method dictionary.
            mode (str): The mode.
            class_method (bool, optional): Whether the method is a class method.
                Defaults to True.

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
        method = getattr(self.__class__, method_name) if class_method else getattr(self, method_name)

        return method, params
