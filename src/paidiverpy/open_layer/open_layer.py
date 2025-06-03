"""Open raw image file."""

import contextlib
import copy
import gc
import logging
import uuid
import dask
import numpy as np
import pandas as pd
from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm
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
        if self.step_name == "raw":
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
                    new_config = copy.copy(self.config)
                    convert_layer = ConvertLayer(
                        config=new_config,
                        metadata=self.metadata,
                        images=self.images,
                        step_name=step_params["name"],
                        parameters=step_params,
                        client=self.client,
                        config_index=None,
                    )
                    self.images = convert_layer.run(add_new_step=False)
                    # remove last step
                    self.config.steps.pop()
                    del convert_layer
                    gc.collect()

    def import_image(self) -> None:
        """Import images with optional Dask parallelization."""
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
                self.set_metadata(
                    SamplingLayer(
                        config=new_config,
                        metadata=self.metadata,
                        parameters=step_params,
                        client=self.client,
                        add_new_step=False,
                    ).run(),
                )
                gc.collect()
                self.config.steps.pop()
        if self.config.is_remote:
            img_path_list = [self.correct_input_path + filename for filename in self.get_metadata()["image-filename"]]
        else:
            img_path_list = [self.correct_input_path / filename for filename in self.get_metadata()["image-filename"]]
        if self.client:
            images_and_exifs = self._process_image_client(img_path_list, remote=self.config.is_remote)
        elif self.n_jobs == 1:
            images_and_exifs = []
            for img_path in tqdm(img_path_list, total=len(img_path_list), desc="Open Images"):
                images_and_exifs.append(self.process_image_sequential(img_path, remote=self.config.is_remote))
        else:
            images_and_exifs = self._process_image_threads(img_path_list, remote=self.config.is_remote)
        exifs, image_list, img_paths = [], [], []
        for img, exif, img_path in images_and_exifs:
            if img is not None:
                image_list.append(img)
                exifs.append(exif)
                img_paths.append(str(img_path).split("/")[-1])
        metadata = self.get_metadata()
        metadata = metadata.loc[metadata["image-filename"].isin(img_paths)]
        with contextlib.suppress(KeyError):
            self.set_metadata(metadata.merge(pd.DataFrame(exifs), on="image-filename", how="left"))
        metadata = self.get_metadata()
        rename = self.step_metadata.get("rename")
        if rename:
            metadata = self.rename_images(rename, metadata)

        self.images.add_step(
            step=self.step_name,
            images=image_list,
            step_metadata=self.step_metadata,
            metadata=metadata,
            track_changes=self.track_changes,
        )
        del image_list
        gc.collect()

    def process_image_sequential(self, img_path: str, remote: bool = False) -> tuple[np.ndarray | dask.array.core.Array, dict, str]:
        """Process a single image file.

        Args:
            img_path (str): The path to the image file
            remote (bool, optional): Whether the image is remote. Defaults to False.

        Returns:
            np.ndarray | dask.array.core.Array, dict, str: The processed image, EXIF data, and image path.
        """
        func = open_image_remote if remote else open_image_local
        img, exif, img_path = func(
            img_path, image_type=self.image_type, image_open_args=self.image_open_args, storage_options=self.storage_options, parallel=False
        )
        return img, exif, img_path

    def _process_image_threads(self, img_path_list: list[str], remote: bool = False) -> list[np.ndarray]:
        """Process images using Dask threads.

        Args:
            img_path_list (list[str]): The list of image paths.
            remote (bool, optional): Whether the images are remote. Defaults to False.

        Returns:
            list[np.ndarray]: The list of processed images.
        """
        func = open_image_remote if remote else open_image_local
        delayed_image_list = []
        for _, img_path in enumerate(img_path_list):
            delayed_image_list.append(
                delayed(func)(
                    img_path, image_type=self.image_type, image_open_args=self.image_open_args, storage_options=self.storage_options, parallel=True
                )
            )
        with dask.config.set(scheduler="threads", num_workers=self.n_jobs):
            with ProgressBar():
                computed_images = compute(*delayed_image_list)
            return list(computed_images)

    def _process_image_client(self, img_path_list: list[str], remote: bool = False) -> list[np.ndarray]:
        """Process images using a Dask client.

        Args:
            img_path_list (list[str]): The list of image paths.
            remote (bool, optional): Whether the images are remote. Defaults to False.

        Returns:
            list[np.ndarray]: The list of processed images.
        """
        func = open_image_remote if remote else open_image_local
        delayed_image_list = []
        if isinstance(self.client.cluster, dask.distributed.LocalCluster):
            for _, img_path in enumerate(img_path_list):
                delayed_image_list.append(
                    delayed(func)(
                        img_path,
                        image_type=self.image_type,
                        image_open_args=self.image_open_args,
                        storage_options=self.storage_options,
                        parallel=True,
                    )
                )
            with ProgressBar():
                futures = self.client.compute(delayed_image_list, sync=False)
        else:
            futures = []
            for _, img_path in enumerate(img_path_list):
                futures.append(
                    self.client.submit(
                        func,
                        img_path,
                        image_type=self.image_type,
                        image_open_args=self.image_open_args,
                        storage_options=self.storage_options,
                        parallel=True,
                    )
                )
        return self.client.gather(futures)

    def rename_images(self, rename: str, metadata: pd.DataFrame) -> pd.DataFrame:
        """Rename images based on the rename mode.

        Args:
            rename (str): The rename mode
            metadata (pd.DataFrame): The metadata

        Raises:
            ValueError: Unknown rename mode

        Returns:
            pd.DataFrame: The renamed metadata
        """
        image_open_args = f".{self.step_metadata.get('image_open_args')}" if self.step_metadata.get("image_open_args") else ""
        if rename == "datetime":
            metadata["image-filename"] = pd.to_datetime(metadata["image-datetime"]).dt.strftime("%Y%m%dT%H%M%S.%f").str[:-3] + "Z" + image_open_args

            duplicate_mask = metadata.duplicated(subset="image-filename", keep=False)
            if duplicate_mask.any():
                duplicates = metadata[duplicate_mask]
                duplicates.loc[:, "duplicate_number"] = duplicates.groupby("image-filename").cumcount() + 1
                metadata.loc[duplicate_mask, "image-filename"] = duplicates.apply(
                    lambda row: f"{row['image-filename'][:-1]}_{row['duplicate_number']}",
                    axis=1,
                )
        elif rename == "UUID":
            metadata["image-filename"] = metadata["image-filename"].apply(lambda _: str(uuid.uuid4()) + image_open_args)
        self.set_metadata(metadata)
        return metadata

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
