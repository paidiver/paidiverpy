"""Main class for the paidiverpy package."""

import logging
from contextlib import suppress
from functools import partial
from pathlib import Path
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from distributed import LocalCluster
from tqdm import tqdm
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.parallellisation import get_client
from paidiverpy.utils.parallellisation import get_n_jobs
import warnings

warnings.simplefilter("always", RuntimeWarning)
warnings.showwarning = lambda *args, **kwargs: __import__("traceback").print_stack()


class Paidiverpy:
    """Main class for the paidiverpy package.

    Args:
        config_params (dict | ConfigParams, optional): The configuration parameters.
            It can contain the following keys / attributes:
            - input_path (str): The path to the input files.
            - output_path (str): The path to the output files.
            - image_open_args (str): The type of the images.
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
        self.track_changes = self.track_changes if track_changes is None else track_changes
        self.layer_methods = None

    def run(self, add_new_step: bool = True) -> ImagesLayer | None:
        """Run the paidiverpy pipeline.

        Args:
            add_new_step (bool, optional): Whether to add a new step. Defaults to True.

        Returns:
            ImagesLayer | None: The images object.
        """
        mode = self.step_metadata.get("mode")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, self.layer_methods, mode)
        images = self.process_images(method, params)
        if not test:
            self.step_name = f"step_{self.config_index}" if not self.step_name else self.step_name
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=images,
                    step_metadata=self.step_metadata,
                    track_changes=self.track_changes,
                )
                self.set_metadata()
                return None
            self.images.replace_step(images=images)
            return self.images
        return None

    # TODO: check if this code is running in parallel correctly
    def process_images(self, method: callable, params: dict, custom: bool = False) -> xr.Dataset:
        """Process the images sequentially.

        Method to process the images sequentially.

        Args:
            method (callable): The method to apply to the images.
            params (dict): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            xr.Dataset: A dataset containing the processed images and the metadata.
        """
        images = self.images.get_step(last=True)

        func = partial(method, params=params)
        processed_images, updated_metadata_list = xr.apply_ufunc(
            Paidiverpy.process_single,
            images["image"],
            images["original_height"],
            images["original_width"],
            # images["mask"],
            images["metadata"],
            input_core_dims=[["y", "x", "band"], [], [], []],
            output_core_dims=[["y", "x", "band"], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[images["image"].dtype, object],
            kwargs={"func": func, "custom": custom},
        )
        if self.client:
            processed_images, updated_metadata_list = self.client.compute([processed_images, updated_metadata_list])
        else:
            processed_images, updated_metadata_list = dask.compute(processed_images, updated_metadata_list)

        processed_images["metadata"] = updated_metadata_list

        return processed_images

    # TODO: update this method
    def process_dataset(
        self,
        images: list[da.core.Array],
        method: callable,
        params: BaseModel,
        custom: bool = False,
    ) -> tuple[list[np.ndarray], pd.DataFrame]:
        """Process the images as a dataset.

        Args:
            images (List[da.core.Array]): The list of images to process.
            method (callable): The method to apply to the images.
            params (BaseModel): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            tuple[list[np.ndarray], pd.DataFrame]: A tuple containing the list of processed images and the metadata DataFrame.
        """
        func = partial(method, params=params)
        metadata = self.get_metadata().to_dict(orient="records")
        if custom:
            processed_images, metadata = func(images, metadata=metadata).process()
        else:
            processed_images, metadata = func(images, metadata=metadata)

        metadata = pd.DataFrame(metadata).set_index("filename")

        return processed_images, metadata

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
            return Configuration(config_file_path=config_file_path)
        general_config = {}
        config_params = ConfigParams(**config_params) if isinstance(config_params, dict) else config_params
        config_params_keys = ["input_path", "output_path", "metadata_path", "metadata_type", "image_open_args", "track_changes", "n_jobs"]
        for key in config_params_keys:
            general_config[key] = getattr(config_params, key)
        return Configuration(add_general=general_config)

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
        metadata = pd.DataFrame(list_of_files, columns=["filename"])
        return metadata.reset_index().rename(columns={"index": "ID"})

    # TODO: check if this code is working properly
    def get_metadata(self, flag: int | None = None, orient: str | None = None) -> pd.DataFrame:
        """Get the metadata object.

        Args:
            flag (int, optional): The flag value. Defaults to None.

        Returns:
            pd.DataFrame: The metadata object.
        """
        flag = 0 if flag is None else flag
        if flag == "all":
            if "image-datetime" not in self.metadata.metadata.columns:
                metadata = self.metadata.metadata.copy()
            else:
                metadata = self.metadata.metadata.sort_values("image-datetime").copy()
        elif "image-datetime" not in self.metadata.metadata.columns:
            metadata = self.metadata.metadata[self.metadata.metadata["flag"] <= flag].copy()
        else:
            metadata = self.metadata.metadata[self.metadata.metadata["flag"] <= flag].sort_values("image-datetime").copy()
        if orient is None:
            return metadata
        return metadata.to_dict(orient=orient)

    def set_metadata(self, image_ds: xr.Dataset | None = None) -> None:
        """Set the metadata.

        Args:
            image_ds (xr.Dataset, optional): The image dataset.
        """
        if image_ds is None:
            image_ds = self.images.get_step(last=True)

        if "metadata" not in image_ds.coords:
            msg = "The dataset does not contain a 'metadata' coordinate."
            raise ValueError(msg)

        metadata_df = pd.DataFrame(list(image_ds["metadata"].values))
        metadata_df["filename"] = image_ds["filename"].to_numpy()
        self.metadata.metadata = metadata_df

    def save_images(
        self,
        step: str | int | None = None,
        image_format: str = "png",
        output_path: str | Path | None = None,
    ) -> None:
        """Save the images.

        Args:
            step (int, optional): The step order. Defaults to None.
            image_format (str, optional): The image format. Defaults to "png".
            output_path (str | Path, optional): The output path. Defaults to None.
        """
        last = False
        if step is None:
            last = True
        if not output_path:
            output_path = self.config.general.output_path
        self.logger.info("Saving images from step: %s", step if not last else "last")

        self.images.save(
            step,
            last=last,
            output_path=output_path,
            image_format=image_format,
            config=self.config,
            client=self.client,
            n_jobs=self.n_jobs,
            logger=self.logger,
        )
        self.metadata.dataset_metadata["output_path"] = str(output_path)
        self.logger.info("Images are saved to: %s", output_path)

    def remove_images(self) -> None:
        """Remove output images from the output path."""
        output_path = self.config.general.output_path
        self.logger.info("Removing images from the output path: %s", output_path)
        self.images.remove(output_path)

    def clear_steps(self, value: int | str) -> None:
        """Clear steps from the images and metadata.

        Args:
            value (int | str): Step name or order.
        """
        self.images.remove_steps_by_order(value)
        self.set_metadata(self.images)

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
        params: BaseModel,
        method_dict: dict,
        mode: str,
        class_method: bool = True,
    ) -> tuple:
        """Get the method by mode.

        Args:
            params (BaseModel): The parameters.
            method_dict (dict): The method dictionary.
            mode (str): The mode.
            class_method (bool, optional): Whether the method is a class method.
                Defaults to True.

        Raises:
            ValueError: Unsupported mode.

        Returns:
            tuple: The method and parameters.
        """
        # if mode not in method_dict:
        #     msg = f"Unsupported mode: {mode}"
        #     raise ValueError(msg)
        method_info = method_dict[mode]
        if not isinstance(params, method_info["params"]):
            params = method_info["params"](**params)
        method_name = method_info["method"]
        method = getattr(self.__class__, method_name) if class_method else getattr(self, method_name)

        return method, params

    def _calculate_raise_error(self) -> bool:
        """Calculate whether to raise an error.

        Returns:
            bool: Whether to raise an error.
        """
        if self.raise_error:
            return self.raise_error
        if isinstance(self.step_metadata["params"], BaseModel):
            raise_error = self.step_metadata["params"].raise_error
        else:
            raise_error = self.step_metadata["params"].get("raise_error", False)
        return raise_error

    # TODO: correct this code
    @staticmethod
    def prepare_inputs(
        image_data: np.ndarray,
        params: BaseModel | None,
        default_params_factory: BaseModel,
        **kwargs: dict,
    ) -> tuple[np.ndarray, dict, BaseModel]:
        """Standard preprocessing for convert layer methods.

        Args:
            image_data (np.ndarray): The image data.
            params (BaseModel | None): The parameters.
            default_params_factory (BaseModel): The default parameters factory.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple[np.ndarray, dict, BaseModel]: The image data, metadata, and parameters.
        """
        _ = kwargs
        params = params or default_params_factory()
        return image_data, params, kwargs

    @staticmethod
    def process_single(img: np.ndarray, height: int, width: int, metadata: dict, func: callable, custom: bool) -> tuple[np.ndarray, dict]:
        """Wrapper to process a single image with its metadata.

        Args:
            img (np.ndarray): The padded image (H, W, bands).
            height (int): The height of the valid image area.
            width (int): The width of the valid image area.
            metadata (dict): The metadata to include.
            func (callable): The processing function.
            custom (bool): Whether to use the custom processing.

        Returns:
            tuple: The processed image (with padding restored) and updated metadata.
        """
        cropped = img[:height, :width, :]

        if custom:
            processed_crop, updated_metadata = func(image_data=cropped, metadata=metadata).process()
        else:
            processed_crop, updated_metadata = func(image_data=cropped, metadata=metadata)

        processed_img = np.full_like(img, 0)
        processed_img[:height, :width, :] = processed_crop

        return processed_img, updated_metadata

    # @staticmethod
    # def process_single(img: np.ndarray, mask: np.ndarray, metadata: dict, func: callable, custom: bool) -> tuple[np.ndarray, dict]:
    #     """Wrapper to process a single image with its metadata.

    #     Args:
    #         img (xr.DataArray or np.ndarray): The image to process.
    #         mask (xr.DataArray or np.ndarray): The mask to apply.
    #         metadata (dict): The metadata to include.
    #         func (callable): The processing function.
    #         custom (bool): Whether to use the custom processing.

    #     Returns:
    #         tuple: The processed image and updated metadata.
    #     """
    #     img = np.where(mask[:, :, np.newaxis] == 1, img, np.nan)

    #     if custom:
    #         processed_img, updated_metadata = func(image_data=img, metadata=metadata).process()
    #     else:
    #         processed_img, updated_metadata = func(image_data=img, metadata=metadata)

    #     return processed_img, updated_metadata
