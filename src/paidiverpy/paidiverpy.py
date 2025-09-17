"""Main class for the paidiverpy package."""

import importlib.util
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.step_config import StepConfigUnion
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.parallellisation import get_n_jobs


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
        config_params: dict[str, Any] | ConfigParams | None = None,
        config_file_path: str | None = None,
        config: Configuration | None = None,
        metadata: MetadataParser | None = None,
        images: ImagesLayer | None = None,
        client: Client | None = None,
        paidiverpy: Optional["Paidiverpy"] = None,
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
            self.images = images or ImagesLayer(
                output_path=self.config.general.output_path,
            )
            self.n_jobs = get_n_jobs(self.config.general.n_jobs)
            self.track_changes = self.config.general.track_changes
            self.client = client
            self.use_dask = bool(self.n_jobs > 1 or self.client is not None)
            self.metadata = metadata or MetadataParser(config=self.config, use_dask=self.use_dask)
        self.track_changes = self.track_changes if track_changes is None else track_changes
        self.layer_methods: dict[str, Any] = {}
        self.step_metadata: dict[str, Any] = {}
        self.config_index: int = 0

    def run(self, add_new_step: bool = True) -> ImagesLayer | None:
        """Run the paidiverpy pipeline.

        Args:
            add_new_step (bool, optional): Whether to add a new step. Defaults to True.

        Returns:
            ImagesLayer | None: The images object.
        """
        mode = self.step_metadata.get("mode", "")
        test = self.step_metadata.get("test", False)
        params: dict[str, Any] | BaseModel = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, self.layer_methods, mode)
        # method, params = self._get_method_by_mode(params, self.layer_methods, mode, False)
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
                return None
            self.images.replace_step(images=images)
            return self.images
        return None

    def process_images(self, method: Callable, params: dict[str, Any] | BaseModel) -> xr.Dataset:
        """Process the images sequentially.

        Method to process the images sequentially.

        Args:
            method (Callable): The method to apply to the images.
            params (dict | BaseModel): The parameters for the method.

        Returns:
            xr.Dataset: A dataset containing the processed images and the metadata.
        """
        images = self.images.get_step(last=True)
        if not images:
            msg = "No images found to process."
            self.logger.error(msg)
            raise_value_error(msg)
        func = partial(method, params=params)

        dask_gufunc_kwargs, output_dtype = self.calculate_output_image(images, func)
        band_name = "new_band" if dask_gufunc_kwargs.get("output_sizes", {}).get("new_band") is not None else "band"
        x_name = "x" if dask_gufunc_kwargs.get("output_sizes", {}).get("new_x") is None else "new_x"
        y_name = "y" if dask_gufunc_kwargs.get("output_sizes", {}).get("new_y") is None else "new_y"

        # meta_img = np.empty((0, 0, 0), dtype=images["images"].dtype)
        # meta_metadata = np.empty((), dtype=object)
        metadata = self.get_metadata().set_index("filename")
        processed_images, original_height, original_width = xr.apply_ufunc(
            Paidiverpy.process_single,
            images["images"],
            images["flag"],
            images["original_height"],
            images["original_width"],
            images["filename"],
            input_core_dims=[["y", "x", "band"], [], [], [], []],
            output_core_dims=[[y_name, x_name, band_name], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[output_dtype, int, int],
            dask_gufunc_kwargs=dask_gufunc_kwargs,
            kwargs={"output_bands": dask_gufunc_kwargs.get("output_sizes", {}).get("new_band"), "func": func, "metadata": metadata},
        )
        processed_images["original_height"].data = original_height.to_numpy()
        processed_images["original_width"].data = original_width.to_numpy()
        if dask_gufunc_kwargs.get("output_sizes", {}).get("new_band") is not None:
            processed_images = processed_images.rename({"new_band": "band"})
            processed_images = processed_images.assign_coords(band=("band", list(range(dask_gufunc_kwargs.get("output_sizes", {}).get("new_band")))))
        if dask_gufunc_kwargs.get("output_sizes", {}).get("new_x") is not None:
            processed_images = processed_images.rename({"new_x": "x"})
            processed_images = processed_images.assign_coords(x=("x", list(range(dask_gufunc_kwargs.get("output_sizes", {}).get("new_x")))))
        if dask_gufunc_kwargs.get("output_sizes", {}).get("new_y") is not None:
            processed_images = processed_images.rename({"new_y": "y"})
            processed_images = processed_images.assign_coords(y=("y", list(range(dask_gufunc_kwargs.get("output_sizes", {}).get("new_y")))))

        return xr.Dataset({"images": processed_images})

    def calculate_output_image(self, images: xr.Dataset, func: Callable) -> tuple[dict[str, Any], np.dtype[Any]]:
        """Calculate the output image dimensions and data type.

        Args:
            images (xr.Dataset): The input images.
            func (Callable): The processing function.

        Returns:
            tuple: A tuple containing the dask_gufunc_kwargs and the output data type.
        """
        flag_zero_mask = images["flag"] == 0
        if not flag_zero_mask.any():
            msg = "No images with flag=0 found for testing output dimensions"
            self.logger.warning(msg)
            raise_value_error(msg)
        test_idx = int(np.argmax(flag_zero_mask.data))

        test_img = images["images"].isel(filename=test_idx).to_numpy()
        test_flag = images["flag"].isel(filename=test_idx).values.item()
        test_height = images["original_height"].isel(filename=test_idx).values.item()
        test_width = images["original_width"].isel(filename=test_idx).values.item()
        test_filename = images["filename"].isel(filename=test_idx).values.item()
        metadata = self.get_metadata().set_index("filename")
        test_result, _, _ = Paidiverpy.process_single(test_img, test_flag, test_height, test_width, test_filename, None, func, metadata)
        output_dtype = test_result.dtype
        output_bands = 1 if test_result.ndim == NUM_DIMENSIONS_GREY else test_result.shape[-1]
        if output_bands == test_img.shape[-1]:
            output_bands = None

        output_x = test_result.shape[1] if test_img.shape[1] != test_result.shape[1] else None
        output_y = test_result.shape[0] if test_img.shape[0] != test_result.shape[0] else None

        dask_gufunc_kwargs: dict[str, Any] = {}
        output_sizes = {}
        if output_x is not None:
            output_sizes["new_x"] = output_x
        if output_y is not None:
            output_sizes["new_y"] = output_y
        if output_bands is not None:
            output_sizes["new_band"] = output_bands

        # Merge into dask_gufunc_kwargs
        if output_sizes:
            dask_gufunc_kwargs["output_sizes"] = {
                **dask_gufunc_kwargs.get("output_sizes", {}),
                **output_sizes,
            }

        return dask_gufunc_kwargs, output_dtype

    def process_dataset(
        self,
        images: xr.Dataset,
        method: Callable,
        params: BaseModel,
    ) -> xr.Dataset:
        """Process the images as a dataset.

        Args:
            images (xr.Dataset): The dataset of images to process.
            method (Callable): The method to apply to the images.
            params (BaseModel): The parameters for the method.

        Returns:
            xr.Dataset: A dataset containing the processed images
        """
        func = partial(method, params=params)
        return func(images)

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
        self.use_dask = paidiverpy.use_dask

    def _initialise_config(
        self,
        config_file_path: str | None,
        config_params: ConfigParams | dict[str, Any] | None,
    ) -> Configuration:
        """Initialize the configuration object.

        Args:
            config_file_path (str | None): Configuration file path.
            config_params (ConfigParams | dict): Configuration parameters.

        Returns:
            Configuration: The configuration object.
        """
        if config_file_path:
            return Configuration(config_file_path=config_file_path)
        general_config: dict[str, Any] = {}
        config_params = ConfigParams(**config_params) if isinstance(config_params, dict) else config_params
        config_params_keys = ["input_path", "output_path", "metadata_path", "metadata_type", "image_open_args", "track_changes", "n_jobs"]
        for key in config_params_keys:
            general_config[key] = getattr(config_params, key)
        return Configuration(add_general=general_config)

    def get_metadata(self, flag: int | str | None = None) -> pd.DataFrame:
        """Get the metadata object.

        Args:
            flag (int | str | None, optional): The flag to filter the metadata.
                If None, return all metadata. If "all", return all metadata sorted by image-datetime.
                Defaults to None.

        Returns:
            pd.DataFrame: The metadata object.
        """
        flag = 0 if flag is None else flag
        metadata = self.metadata.metadata
        if flag == "all":
            if "image-datetime" not in metadata.columns:
                return metadata.copy()
            return metadata.sort_values("image-datetime").copy()
        if "image-datetime" not in metadata.columns:
            return metadata[metadata["flag"] <= flag].copy()
        return metadata[metadata["flag"] <= flag].sort_values("image-datetime").copy()

    def set_metadata(self, metadata: pd.DataFrame | None = None, dataset_metadata: dict[str, Any] | None = None) -> None:
        """Set the metadata.

        Args:
            metadata (pd.DataFrame | None): The metadata to set.
            dataset_metadata (dict | None): The dataset metadata to set.
        """
        self.metadata.set_metadata(metadata, dataset_metadata)

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
            config=self.config,
            step=step,
            last=last,
            output_path=output_path,
            image_format=image_format,
            client=self.client,
            n_jobs=self.n_jobs,
            use_dask=self.use_dask,
        )
        self.set_metadata(dataset_metadata={"output_path": str(output_path)})
        self.logger.info("Images are saved to the path: %s", output_path)

    def remove_images(self) -> None:
        """Remove output images from the output path."""
        output_path = self.config.general.output_path
        self.logger.info("Removing images from the output path: %s", output_path)
        self.images.remove(output_path)

    def _calculate_steps_metadata(self, config_part: StepConfigUnion) -> dict[str, object]:
        """Calculate the steps metadata.

        Args:
            config_part (StepConfigUnion): The configuration part.

        Returns:
            dict: The steps metadata.
        """
        return dict(config_part.__dict__.items())

    def _get_method_by_mode(
        self,
        params: BaseModel | dict[str, Any],
        method_dict: dict[str, Any],
        mode: str,
        class_method: bool = True,
    ) -> tuple[Callable, BaseModel]:
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

    def _merge_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Merge the metadata with the new metadata.

        Args:
            metadata (pd.DataFrame): The original metadata.

        Returns:
            pd.DataFrame: The merged metadata.
        """
        new_meta = self.get_metadata(flag="all").set_index("filename")
        meta = metadata.set_index("filename")

        for col in meta.columns:
            if col in new_meta:
                new_meta.update(meta[[col]])
            else:
                new_meta[col] = meta[col]

        return new_meta.reset_index()

    def load_custom_algorithm(self, file_path: str, class_name: str, algorithm_name: str) -> Callable:
        """Load a custom algorithm class.

        Args:
            file_path (str): The file path of the custom algorithm.
            class_name (str): The class name.
            algorithm_name (str): The algorithm name.

        Returns:
            class: The custom algorithm class.
        """
        spec = importlib.util.spec_from_file_location(algorithm_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, class_name)

    @staticmethod
    def process_single(
        img: np.ndarray[Any, Any],
        flag: int,
        height: int,
        width: int,
        filename: str,
        output_bands: int | None,
        func: Callable,
        metadata: pd.DataFrame,
    ) -> tuple[np.ndarray[Any, Any], int, int]:
        """Wrapper to process a single image with its metadata.

        Args:
            img (np.ndarray): The padded image (H, W, bands).
            flag (int): The flag indicating the processing step.
            height (int): The height of the valid image area.
            width (int): The width of the valid image area.
            filename (str): The filename of the image.
            output_bands (int): The number of output bands.
            func (Callable): The processing function.
            metadata (pd.DataFrame): The metadata DataFrame.

        Returns:
            tuple: A tuple containing the processed image, height, and width.
        """
        if flag > 0:
            if output_bands is None:
                output_bands = img.shape[-1]
            return np.zeros((img.shape[0], img.shape[1], output_bands), dtype=img.dtype), height, width
        cropped = img[:height, :width, :]

        processed_crop = func(image_data=cropped, filename=filename, metadata=metadata)
        if processed_crop.ndim == NUM_DIMENSIONS_GREY:
            processed_crop = np.expand_dims(processed_crop, axis=-1)
        if output_bands is not None and output_bands != processed_crop.shape[-1]:
            return np.zeros((img.shape[0], img.shape[1], output_bands), dtype=img.dtype), height, width
        if processed_crop.shape[0] != height:
            height = processed_crop.shape[0]
        if processed_crop.shape[1] != width:
            width = processed_crop.shape[1]

        final_height = max(height, img.shape[0])
        final_width = max(width, img.shape[1])
        processed_img = np.zeros((final_height, final_width, processed_crop.shape[-1]), dtype=processed_crop.dtype)
        processed_img[:height, :width, : processed_crop.shape[-1]] = processed_crop

        return processed_img, height, width
