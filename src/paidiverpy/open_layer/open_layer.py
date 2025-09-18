"""Open raw image file."""

import gc
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.open_params import SUPPORTED_OPENCV_IMAGE_TYPES
from paidiverpy.models.open_params import SUPPORTED_RAWPY_IMAGE_TYPES
from paidiverpy.models.open_params import ImageOpenArgsOpenCVParams
from paidiverpy.models.open_params import ImageOpenArgsRawParams
from paidiverpy.models.open_params import ImageOpenArgsRawPyParams
from paidiverpy.open_layer.utils import open_image_local
from paidiverpy.open_layer.utils import open_image_remote
from paidiverpy.open_layer.utils import pad_image
from paidiverpy.sampling_layer import SamplingLayer
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.object_store import define_storage_options


class OpenLayer(Paidiverpy):
    """Open raw image file.

    Args:
        config_params (dict | ConfigParams, optional): The configuration parameters.
            It can contain the following keys / attributes:
            - input_path (str): The path to the input files.
            - output_path (str): The path to the output files.
            - metadata_path (str): The path to the metadata file.
            - metadata_type (str): The type of the metadata file.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of n_jobs.
        config_file_path (str): The path to the configuration file.
        config (Configuration): The configuration object.
        metadata (MetadataParser): The metadata object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        step_name (str): The name of the step.
        parameters (dict): The parameters for the step.
        client (Client): The Dask client.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        config_params: dict[str, Any] | ConfigParams | None = None,
        config_file_path: str | None = None,
        config: Configuration | None = None,
        metadata: MetadataParser | None = None,
        images: ImagesLayer | None = None,
        paidiverpy: Paidiverpy | None = None,
        step_name: str = "raw",
        client: Client | None = None,
        parameters: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        super().__init__(
            config_params=config_params,
            config_file_path=config_file_path,
            metadata=metadata,
            config=config,
            images=images,
            paidiverpy=paidiverpy,
            client=client,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )
        self.step_name = step_name
        if parameters:
            self.config.add_general(parameters)
        is_docker = is_running_in_docker()
        self.storage_options = define_storage_options(self.config.general.input_path)

        if self.config.general.sample_data or self.config.is_remote:
            self.correct_input_path = self.config.general.input_path
        else:
            self.correct_input_path = "/app/input/" if is_docker else self.config.general.input_path
        self.step_metadata = self._calculate_steps_metadata(self.config.general)
        self.image_type, self.image_open_args = self._get_image_open_args(self.config.general.image_open_args)

    def run(self) -> None:
        """Run the open layer steps based on the configuration file or parameters."""
        self.import_image()
        if self.step_metadata.get("convert"):
            for step in self.step_metadata.get("convert"):
                dict_step = step.to_dict() if issubclass(type(step), BaseModel) else step
                step_params = {
                    "step_name": "convert",
                    "name": dict_step.get("mode"),
                    "mode": dict_step.get("mode"),
                    "params": dict_step.get("params"),
                }
                # new_config = copy.copy(self.config)
                self.images = ConvertLayer(
                    paidiverpy=self,
                    step_name=step_params["name"],
                    parameters=step_params,
                    config_index=None,
                ).run(add_new_step=False)
                self.config.steps.pop()
                gc.collect()

    def import_image(self) -> None:
        """Import images with optional Dask parallelization."""
        metadata = None
        if self.step_metadata.get("sampling"):
            for step in self.step_metadata.get("sampling"):
                dict_step = step.to_dict() if issubclass(type(step), BaseModel) else step
                step_params = {
                    "step_name": "sampling",
                    "name": dict_step.get("mode"),
                    "mode": dict_step.get("mode"),
                    "params": dict_step.get("params"),
                }
                metadata = SamplingLayer(
                    paidiverpy=self,
                    parameters=step_params,
                    config_index=None,
                ).run(add_new_step=False)
                gc.collect()
                self.config.steps.pop()

        if metadata is None:
            metadata = self.get_metadata()
        images_info = self._process_images(metadata=metadata, remote=self.config.is_remote)
        image_ds = self.create_dataset(images_info)
        self.images.add_step(
            step=self.step_name,
            images=image_ds,
            step_metadata=self.step_metadata,
            track_changes=self.track_changes,
        )
        gc.collect()

    def _process_images(self, metadata: pd.DataFrame, remote: bool = False) -> dict[str, Any]:
        """Process images using Dask threads.

        Args:
            metadata (pd.DataFrame): The metadata DataArray.
            remote (bool, optional): Whether the images are remote. Defaults to False.

        Returns:
            dict: A dictionary containing processed image information.
        """
        func = open_image_remote if remote else open_image_local
        images_info: dict[str, list[Any]] = {"image": [], "metadata": [], "height": [], "width": []}
        metadata = metadata.set_index("filename")
        rename = self.step_metadata.get("rename")
        if self.use_dask:
            delayed_list = []
            for filename, file_metadata in metadata.iterrows():
                img_path = self.correct_input_path + filename if remote else self.correct_input_path / filename
                delayed_list.append(
                    delayed(OpenLayer.process_single_image)(
                        img_path=img_path,
                        func=func,
                        metadata=file_metadata,
                        rename=rename,
                        image_type=self.image_type,
                        image_open_args=self.image_open_args,
                        storage_options=self.storage_options,
                    )
                )

            if self.client is not None:
                results = self.client.gather(self.client.compute(delayed_list))
            else:
                with dask.config.set(scheduler="threads", num_workers=self.n_jobs), ProgressBar():
                    results = list(compute(*delayed_list))
            for result in results:
                images_info["image"].append(result[0])
                images_info["metadata"].append(result[1])
                images_info["height"].append(result[2])
                images_info["width"].append(result[3])
        else:
            for filename, file_metadata in metadata.iterrows():
                img_path = self.correct_input_path + filename if remote else self.correct_input_path / filename
                img, metadata, height, width = OpenLayer.process_single_image(
                    img_path=img_path,
                    func=func,
                    metadata=file_metadata,
                    rename=rename,
                    image_type=self.image_type,
                    image_open_args=self.image_open_args,
                    storage_options=self.storage_options,
                )
                images_info["image"].append(img)
                images_info["metadata"].append(metadata)
                images_info["height"].append(height)
                images_info["width"].append(width)
        return images_info

    @staticmethod
    def process_single_image(
        img_path: str | Path,
        func: Callable,
        metadata: xr.DataArray,
        rename: str,
        image_type: str,
        image_open_args: dict[str, Any],
        storage_options: dict[str, Any],
    ) -> tuple[np.ndarray[Any, Any] | da.core.Array, dict[str, Any], str] | None:
        """Process a single image.

        Args:
            img_path (str | Path): The path to the image.
            func (Callable): The function to process the image.
            metadata (xr.DataArray): The metadata DataArray.
            rename (str): The rename strategy.
            image_type (str): The image type.
            image_open_args (dict): The image open arguments.
            storage_options (dict): The storage options.
        """
        img, exif, filename = func(img_path=img_path, image_type=image_type, image_open_args=image_open_args, storage_options=storage_options)
        if img is None:
            return None
        height = img.shape[0] if img is not None else 0
        width = img.shape[1] if img is not None else 0
        new_filename = filename
        if rename == "datetime":
            new_filename = metadata["image-datetime"].isoformat()
        elif rename == "UUID":
            new_filename = str(uuid.uuid4())
        metadata["new_filename"] = new_filename
        if exif:
            metadata.update(exif)
        return img, metadata, height, width

    def create_dataset(self, images_info: dict[str, Any]) -> xr.Dataset:
        """Create a Dask array from the processed images and EXIF data.

        Args:
            images_info (dict): A dictionary containing processed image information.

        Returns:
            xr.Dataset: The image dataset.
        """
        metadata = pd.DataFrame(images_info["metadata"])
        metadata = metadata.reset_index(drop=True)
        metadata = metadata.rename(columns={"new_filename": "filename"})
        counts = metadata.groupby("filename").cumcount()
        metadata["filename"] = metadata["filename"] + counts.replace(0, "").astype(str)

        max_height = max(images_info["height"])
        max_width = max(images_info["width"])
        image_list: list[np.ndarray[Any, Any] | da.core.Array] = []
        for img in images_info["image"]:
            padded = pad_image(img, max_height, max_width)
            image_list.append(padded)

        stacked_imgs = np.stack(image_list, axis=0)

        output_ds = xr.Dataset(
            data_vars={"images": (["filename", "y", "x", "band"], stacked_imgs)},
            coords={
                "filename": np.array(metadata["filename"], dtype=str),
                "y": np.arange(max_height),
                "x": np.arange(max_width),
                "band": np.arange(stacked_imgs.shape[-1]),
                # "metadata": (["filename"], new_metadata),
                "original_height": (["filename"], images_info["height"]),
                # "original_bands": (["filename"], band),
                "original_width": (["filename"], images_info["width"]),
                "flag": (["filename"], metadata["flag"]),
            },
            attrs={
                "description": "Image Dataset",
            },
        )
        if self.use_dask:
            # metadata = dd.from_pandas(pd.DataFrame(metadata), npartitions=1)
            output_ds = output_ds.chunk({"filename": 1})
        self.set_metadata(metadata=metadata)
        return output_ds

    def _get_image_open_args(self, image_open_args: str | dict[str, Any] | ConfigParams) -> tuple[str | None, str | dict[str, Any]]:
        """Get the image open arguments.

        Args:
            image_open_args (str | dict | None): The image open arguments

        Returns:
            tuple[str | None, str | None]: The image type and parameters
        """
        if isinstance(image_open_args, str):
            image_type = image_open_args.lower()
            image_open_args = self._define_image_open_args(image_type, {})
        elif isinstance(image_open_args, ConfigParams):
            image_open_args = image_open_args.to_dict()
            image_type = image_open_args["image_type"].lower()
            image_open_args = image_open_args["params"]
        else:
            image_type = image_open_args["image_type"].lower()
            image_open_args = self._define_image_open_args(image_type, image_open_args.get("params", {}))
        return image_type, image_open_args

    def _define_image_open_args(self, image_type: str, params: dict[str, Any]) -> str | dict[str, Any]:
        """Define the image open arguments based on the image type.

        Args:
            image_type (str): The image type
            params (dict): The parameters

        Returns:
            str | dict: The image open arguments
        """
        if image_type in SUPPORTED_RAWPY_IMAGE_TYPES:
            image_open_args = ImageOpenArgsRawPyParams(**params)
        elif image_type in SUPPORTED_OPENCV_IMAGE_TYPES:
            image_open_args = ImageOpenArgsOpenCVParams(**params)
        else:
            image_open_args = ImageOpenArgsRawParams(**params)
        return image_open_args.to_dict()
