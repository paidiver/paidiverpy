"""Open raw image file."""

import copy
import gc
import logging
import uuid
from io import BytesIO
import cv2
import dask
import dask.array as da
import dask_image.imread
import numpy as np
import pandas as pd
from dask import compute
from dask import delayed
from dask.diagnostics import ProgressBar
from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.resample_layer import ResampleLayer
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.dynamic_classes import DynamicConfig
from paidiverpy.utils.object_store import define_storage_options
from paidiverpy.utils.object_store import get_file_from_bucket


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
            self.config.add_config("general", parameters)
        is_docker = is_running_in_docker()
        self.storage_options = define_storage_options(self.config.general.input_path)

        if self.config.general.sample_data or self.config.general.is_remote:
            self.correct_input_path = self.config.general.input_path
        else:
            self.correct_input_path = "/app/input/" if is_docker else self.config.general.input_path
        self.step_metadata = self._calculate_steps_metadata(self.config.general)

    def run(self) -> None:
        """Run the open layer steps based on the configuration file or parameters."""
        if self.step_name == "raw":
            self.import_image()
            if self.step_metadata.get("convert"):
                for step in self.step_metadata.get("convert"):
                    dict_step = step.to_dict() if issubclass(type(step), DynamicConfig) else step
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
                dict_step = step.to_dict() if issubclass(type(step), DynamicConfig) else step
                step_params = {
                    "step_name": "sampling",
                    "name": dict_step.get("mode"),
                    "mode": dict_step.get("mode"),
                    "params": dict_step.get("params"),
                }
                new_config = copy.copy(self.config)
                self.set_metadata(
                    ResampleLayer(
                        config=new_config,
                        metadata=self.metadata,
                        parameters=step_params,
                        client=self.client,
                    ).run(),
                )
                gc.collect()
                self.config.steps.pop()
        if self.config.general.is_remote:
            img_path_list = [self.correct_input_path + filename for filename in self.get_metadata()["image-filename"]]
        else:
            img_path_list = [self.correct_input_path / filename for filename in self.get_metadata()["image-filename"]]
        if self.client:
            images_and_exifs = self._process_image_client(img_path_list, remote=self.config.general.is_remote)
        elif self.n_jobs == 1:
            images_and_exifs = []
            for img_path in tqdm(img_path_list, total=len(img_path_list), desc="Open Images"):
                images_and_exifs.append(self.process_image_sequential(img_path, remote=self.config.general.is_remote))
        else:
            images_and_exifs = self._process_image_threads(img_path_list, remote=self.config.general.is_remote)
        exifs, image_list = [], []
        for img, exif in images_and_exifs:
            image_list.append(img)
            exifs.append(exif)

        self.set_metadata(self.get_metadata().merge(pd.DataFrame(exifs), on="image-filename", how="left"))
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

    def process_image_sequential(self, img_path: str, remote: bool = False) -> tuple[np.ndarray | dask.array.core.Array, dict]:
        """Process a single image file.

        Args:
            img_path (str): The path to the image file
            remote (bool, optional): Whether the image is remote. Defaults to False.

        Returns:
            np.ndarray | dask.array.core.Array: The processed image data
        """
        func = OpenLayer.open_image_remote if remote else OpenLayer.open_image_local
        img, exif = func(img_path, storage_options=self.storage_options, parallel=False)
        return img, exif

    def _process_image_threads(self, img_path_list: list[str], remote: bool = False) -> list[np.ndarray]:
        """Process images using Dask threads.

        Args:
            img_path_list (list[str]): The list of image paths.
            remote (bool, optional): Whether the images are remote. Defaults to False.

        Returns:
            list[np.ndarray]: The list of processed images.
        """
        func = OpenLayer.open_image_remote if remote else OpenLayer.open_image_local
        delayed_image_list = []
        for _, img_path in enumerate(img_path_list):
            delayed_image_list.append(delayed(func)(img_path, storage_options=self.storage_options, parallel=True))
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
        func = OpenLayer.open_image_remote if remote else OpenLayer.open_image_local
        delayed_image_list = []
        if isinstance(self.client.cluster, dask.distributed.LocalCluster):
            for _, img_path in enumerate(img_path_list):
                delayed_image_list.append(delayed(func)(img_path, storage_options=self.storage_options, parallel=True))
            with ProgressBar():
                futures = self.client.compute(delayed_image_list, sync=False)
        else:
            futures = []
            for _, img_path in enumerate(img_path_list):
                futures.append(self.client.submit(func, img_path, storage_options=self.storage_options, parallel=True))
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
        image_type = f".{self.step_metadata.get('image_type')}" if self.step_metadata.get("image_type") else ""
        if rename == "datetime":
            metadata["image-filename"] = pd.to_datetime(metadata["image-datetime"]).dt.strftime("%Y%m%dT%H%M%S.%f").str[:-3] + "Z" + image_type

            duplicate_mask = metadata.duplicated(subset="image-filename", keep=False)
            if duplicate_mask.any():
                duplicates = metadata[duplicate_mask]
                duplicates.loc[:, "duplicate_number"] = duplicates.groupby("image-filename").cumcount() + 1
                metadata.loc[duplicate_mask, "image-filename"] = duplicates.apply(
                    lambda row: f"{row['image-filename'][:-1]}_{row['duplicate_number']}",
                    axis=1,
                )
        elif rename == "UUID":
            metadata["image-filename"] = metadata["image-filename"].apply(lambda _: str(uuid.uuid4()) + image_type)
        else:
            self.logger.error("Unknown rename mode: %s", rename)
            if self.raise_error:
                msg = f"Unknown rename mode: {rename}"
                raise ValueError(msg)
        self.set_metadata(metadata)
        return metadata

    @staticmethod
    def open_image_remote(img_path: str, **kwargs: dict) -> tuple[np.ndarray | dask.array.core.Array, dict]:
        """Open an image file.

        Args:
            img_path (str): The path to the image file
            **kwargs (dict): Additional keyword arguments. The following are supported:
                - storage_options (dict): The storage options for reading metadata file.
                - parallel (bool): Whether to use Dask for parallel processing.

        Raises:
            ValueError: Failed to open the image

        Returns:
            tuple[np.ndarray | dask.array.core.Array, dict]: The image data and the EXIF data
        """
        try:
            img_bytes = get_file_from_bucket(img_path, kwargs.get("storage_options"))
            if kwargs.get("parallel"):
                img_array = np.frombuffer(img_bytes, np.uint8)
                decoded_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                lazy_img = delayed(cv2.imdecode)(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                img = da.from_delayed(lazy_img, shape=decoded_img.shape, dtype=decoded_img.dtype)
            else:
                img_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            exif = OpenLayer.extract_exif_single(BytesIO(img_bytes), image_name=img_path.split("/")[-1])
        except (FileNotFoundError, OSError, TypeError) as e:
            img = None
            logging.warning("Failed to open %s: %s", img_path, e)

        return img, exif

    @staticmethod
    def open_image_local(img_path: str, **kwargs: dict) -> tuple[np.ndarray | dask.array.core.Array, dict]:
        """Open an image file.

        Args:
            img_path (str): The path to the image file
            **kwargs (dict): Additional keyword arguments. The following are supported:
                - parallel (bool): Whether to use Dask for parallel processing.

        Raises:
            ValueError: Failed to open the image

        Returns:
            tuple[np.ndarray | dask.array.core.Array, dict]: The image data and the EXIF data
        """
        exif = OpenLayer.extract_exif_single(img_path)
        if kwargs.get("parallel"):
            img = dask_image.imread.imread(str(img_path))
            img = np.squeeze(img)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        return img, exif

    @staticmethod
    def extract_exif_single(img_path: str, image_name: str | None = None) -> dict:
        """Extract EXIF data from a single image file.

        Args:
            img_path (str): The path to the image file.
            image_name (str, optional): The name of the image file. Defaults to None.

        Returns:
            dict: The EXIF data.
        """
        exif = {}
        try:
            img_pil = Image.open(img_path)
            exif_data = img_pil.getexif()
            if exif_data is not None:
                if image_name:
                    exif["image-filename"] = image_name
                else:
                    exif["image-filename"] = img_path.name
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    exif[tag_name] = value
        except FileNotFoundError as e:
            logging.warning("Failed to open %s: %s", img_path, e)
        except OSError as e:
            logging.warning("Failed to open %s: %s", img_path, e)
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to extract EXIF data from %s: %s", img_path, e)
        return exif
