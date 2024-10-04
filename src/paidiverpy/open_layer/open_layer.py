"""Open raw image file."""

import copy
import gc
import logging
import uuid
import cv2
import dask
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
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.resample_layer import ResampleLayer
from utils import DynamicConfig


class OpenLayer(Paidiverpy):
    """Open raw image file.

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
        step_name (str): The name of the step.
        parameters (dict): The parameters for the step.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
        track_changes (bool): Whether to track changes. Defaults to True.
        n_jobs (int): The number of jobs to run in parallel.
    """

    def __init__(
        self,
        config_file_path: str | None=None,
        input_path: str | None=None,
        output_path: str | None=None,
        metadata_path: str | None=None,
        metadata_type: str | None=None,
        metadata: MetadataParser = None,
        config: Configuration = None,
        logger: logging.Logger | None = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str = "raw",
        parameters: dict | None = None,
        raise_error: bool = False,
        verbose: int = 2,
        track_changes: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata_type=metadata_type,
            metadata=metadata,
            config=config,
            logger=logger,
            images=images,
            paidiverpy=paidiverpy,
            raise_error=raise_error,
            verbose=verbose,
            track_changes=track_changes,
            n_jobs=n_jobs,
        )

        self.step_name = step_name
        if parameters:
            self.config.add_config("general", parameters)

        self.extract_exif()
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
                    ).run(),
                )
                self.config.steps.pop()

        img_path_list = [
            self.config.general.input_path / filename for filename in self.get_metadata()["image-filename"]
        ]
        if self.n_jobs == 1:
            image_list = [
                self.process_image(img_path)
                for img_path in tqdm(img_path_list, total=len(img_path_list), desc="Open Images")
            ]
        else:
            delayed_image_list = [delayed(self.process_image)(img_path) for _, img_path in enumerate(img_path_list)]
            with dask.config.set(scheduler="threads", num_workers=self.n_jobs):
                self.logger.info("Processing images using %s cores", self.n_jobs)
                with ProgressBar():
                    computed_images = compute(*delayed_image_list)
                image_list = list(computed_images)

        metadata = self.get_metadata()
        rename = self.step_metadata.get("rename")
        if rename:
            image_type = f".{self.step_metadata.get('image_type')}" if self.step_metadata.get("image_type") else ""
            if rename == "datetime":
                metadata["image-filename"] = (
                    pd.to_datetime(metadata["image-datetime"]).dt.strftime("%Y%m%dT%H%M%S.%f").str[:-3]
                    + "Z"
                    + image_type
                )

                duplicate_mask = metadata.duplicated(subset="image-filename", keep=False)
                if duplicate_mask.any():
                    duplicates = metadata[duplicate_mask]
                    duplicates["duplicate_number"] = duplicates.groupby("image-filename").cumcount() + 1
                    metadata.loc[duplicate_mask, "image-filename"] = duplicates.apply(
                        lambda row: f"{row['image-filename'][:-1]}_{row['duplicate_number']}",
                        axis=1,
                    )
            elif rename == "UUID":
                metadata["image-filename"] = metadata["image-filename"].apply(
                    lambda _: str(uuid.uuid4()) + image_type)
            else:
                self.logger.error("Unknown rename mode: %s", rename)
                if self.raise_error:
                    msg = f"Unknown rename mode: {rename}"
                    raise ValueError(msg)
            self.set_metadata(metadata)

        self.images.add_step(
            step=self.step_name,
            images=image_list,
            step_metadata=self.step_metadata,
            metadata=metadata,
            track_changes=self.track_changes,
        )
        del image_list
        gc.collect()

    def process_image(self, img_path: str) -> np.ndarray | dask.array.core.Array:
        """Process a single image file.

        Args:
            img_path (str): The path to the image file

        Returns:
            Union[np.ndarray, dask.array.core.Array]: The processed image data
        """
        img = self.open_image(img_path=img_path)
        gc.collect()
        return img

    def open_image(self, img_path: str) -> np.ndarray | dask.array.core.Array:
        """Open an image file.

        Args:
            img_path (str): The path to the image file

        Raises:
            ValueError: Failed to open the image

        Returns:
            Union[np.ndarray, dask.array.core.Array]: The image data
        """
        if self.n_jobs == 1:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            img = dask_image.imread.imread(img_path)
            img = np.squeeze(img)
        if img is None:
            if self.raise_error:
                self.logger.error("Failed to open the image: %s", img_path)
                msg = "Failed to open the image"
                raise ValueError(msg)
            self.logger.warning("Failed to open the image: %s", img_path)
            return None
        return img

    def extract_exif(self) -> None:
        """Extract EXIF data from the images and add it to the metadata DataFrame."""
        img_path_list = [
            self.config.general.input_path / filename for filename in self.get_metadata()["image-filename"]
        ]
        exif_list = [OpenLayer.extract_exif_single(img_path,
                                                   self.logger) for img_path in img_path_list]
        self.set_metadata(self.get_metadata(flag="all").merge(pd.DataFrame(exif_list), on="image-filename", how="left"))

    @staticmethod
    def extract_exif_single(img_path: str, logger: logging.Logger) -> dict:
        """Extract EXIF data from a single image file.

        Args:
            img_path (str): The path to the image file.
            logger (logging.Logger): The logger object.

        Returns:
            dict: The EXIF data.
        """
        exif = {}
        try:
            img_pil = Image.open(img_path)
            exif_data = img_pil.getexif()
            if exif_data is not None:
                exif["image-filename"] = img_path.name
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    exif[tag_name] = value
        except FileNotFoundError as e:
            logger.warning("Failed to open %s: %s", img_path, e)
        except OSError as e:
            logger.warning("Failed to open %s: %s", img_path, e)
        except Exception as e:
            logger.warning("Failed to extract EXIF data from %s: %s", img_path, e)
        return exif
