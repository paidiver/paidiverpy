"""Open raw image file."""

import copy
import gc
import logging
import uuid
import dask
import numpy as np
import xarray as xr
from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
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
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str = "raw",
        parameters: dict | None = None,
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
                new_config = copy.copy(self.config)
                metadata = SamplingLayer(
                    config=new_config,
                    metadata=self.metadata,
                    parameters=step_params,
                    client=self.client,
                    add_new_step=False,
                ).run()
                gc.collect()
                self.config.steps.pop()

        if metadata is None:
            metadata = self.get_metadata()

        if self.config.is_remote:
            img_path_list = [self.correct_input_path + filename.item() for filename in metadata["filename"]]
        else:
            img_path_list = [self.correct_input_path / filename.item() for filename in metadata["filename"]]
        images_and_exifs = self._process_images(img_path_list, remote=self.config.is_remote)
        image_ds = self.create_dataset(images_and_exifs, metadata)
        self.images.add_step(
            step=self.step_name,
            images=image_ds,
            step_metadata=self.step_metadata,
            track_changes=self.track_changes,
        )
        self.metadata.metadata = None
        self.metadata.dataset_metadata = {}
        gc.collect()

    def _process_images(self, img_path_list: list[str], remote: bool = False) -> list[np.ndarray]:
        """Process images using Dask threads.

        Args:
            img_path_list (list[str]): The list of image paths.
            remote (bool, optional): Whether the images are remote. Defaults to False.

        Returns:
            list[np.ndarray]: The list of processed images.
        """
        func = open_image_remote if remote else open_image_local
        delayed_list = [
            delayed(func)(path, image_type=self.image_type, image_open_args=self.image_open_args, storage_options=self.storage_options, parallel=True)
            for path in img_path_list
        ]

        if self.client:
            if isinstance(self.client.cluster, dask.distributed.LocalCluster):
                # Local cluster
                with ProgressBar():
                    futures = self.client.compute(delayed_list, sync=False)
            else:
                # Remote/distributed cluster
                futures = [self.client.submit(d, pure=False) for d in delayed_list]
            return self.client.gather(futures)
        with dask.config.set(scheduler="threads", num_workers=self.n_jobs), ProgressBar():
            return list(compute(*delayed_list))

    def create_dataset(self, images_and_exifs: list[tuple[np.ndarray | dask.array.core.Array, dict, str]], metadata: xr.DataArray) -> xr.Dataset:
        """Create a Dask array from the processed images and EXIF data.

        Args:
            images_and_exifs (list[tuple[np.ndarray | dask.array.core.Array, dict, str]]): The list of processed images and EXIF data.
            metadata (xr.DataArray): The metadata DataArray.

        Returns:
            xr.Dataset: The image dataset.
        """
        rename = self.step_metadata.get("rename")
        seen = {}

        image_list, img_paths, height, width, band, new_metadata, flags = [], [], [], [], [], [], []
        for img, exif, img_path in images_and_exifs:
            if img is not None:
                image_list.append(img)
                height.append(img.shape[0])
                width.append(img.shape[1])
                band.append(img.shape[-1])
                filename = str(img_path).split("/")[-1]
                local_metadata = metadata.loc[{"filename": filename}].item()
                new_filename = filename
                if rename == "datetime":
                    new_filename = local_metadata["image-datetime"].isoformat()
                    if new_filename in seen:
                        seen[new_filename] += 1
                        new_filename = f"{new_filename}_{seen[new_filename]}"
                    else:
                        seen[new_filename] = 1
                elif rename == "UUID":
                    new_filename = str(uuid.uuid4())

                if exif:
                    local_metadata.update(exif)
                local_metadata["filename"] = new_filename
                img_paths.append(new_filename)

                new_metadata.append(local_metadata)
                flags.append(local_metadata.get("flag", 0))

        max_height = max(height)
        max_width = max(width)
        new_image_list, masks = [], []
        for img in image_list:
            padded, mask = pad_image(img, max_height, max_width)
            new_image_list.append(padded)
            masks.append(mask)

        stacked_imgs = np.stack(new_image_list, axis=0)
        return xr.Dataset(
            data_vars={"images": (["filename", "y", "x", "band"], stacked_imgs)},
            coords={
                "filename": img_paths,
                "y": np.arange(max_height),
                "x": np.arange(max_width),
                "band": np.arange(stacked_imgs.shape[-1]),
                "metadata": (["filename"], new_metadata),
                "original_height": (["filename"], height),
                # "original_bands": (["filename"], band),
                "original_width": (["filename"], width),
                "flag": (["filename"], flags),
            },
            attrs={
                "description": "Image Dataset",
                "dataset_metadata": metadata.attrs.get("dataset_metadata", {}),
            },
        )

    def _get_image_open_args(self, image_open_args: str | dict | None) -> tuple[str | None, str | None]:
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

    def _define_image_open_args(self, image_type: str, params: dict) -> str | dict:
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
