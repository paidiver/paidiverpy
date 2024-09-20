""" Open raw image file
"""

import gc
import copy
from typing import Union

from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import dask
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask_image.imread

from paidiverpy import Paidiverpy
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.image_layer import ImageLayer
from paidiverpy.resample_layer import ResampleLayer
from utils import DynamicConfig

class OpenLayer(Paidiverpy):
    def __init__(
        self,
        config_file_path=None,
        input_path=None,
        output_path=None,
        catalog_path=None,
        catalog_type=None,
        catalog=None,
        config=None,
        logger=None,
        images=None,
        paidiverpy=None,
        step_name="raw",
        parameters=None,
        raise_error=False,
        verbose=True,
        n_jobs: int =1,
    ):
        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            catalog_path=catalog_path,
            catalog_type=catalog_type,
            catalog=catalog,
            config=config,
            logger=logger,
            images=images,
            paidiverpy=paidiverpy,
            raise_error=raise_error,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        self.step_name = step_name
        if parameters:
            self.config.add_config("general", parameters)

        self.extract_exif()
        self.step_metadata = self._calculate_steps_metadata(self.config.general)

    def run(self):
        if self.step_name == "raw":
            self.import_image()
            if self.step_metadata.get("convert"):
                for step in self.step_metadata.get("convert"):
                    if issubclass(type(step), DynamicConfig):
                        step = step.to_dict()
                    step_params = {
                        "step_name": "convert",
                        "name": step.get("mode"),
                        "mode": step.get("mode"),
                        "params": step.get("params"),
                    }
                    new_config = copy.copy(self.config)
                    convert_layer = ConvertLayer(
                        config=new_config,
                        catalog=self.catalog,
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

    def import_image(self):
        """Import images with optional Dask parallelization"""
        if self.step_metadata.get("sampling"):
            for step in self.step_metadata.get("sampling"):
                if issubclass(type(step), DynamicConfig):
                    step = step.to_dict()
                step_params = {
                    "step_name": "sampling",
                    "name": step.get("mode"),
                    "mode": step.get("mode"),
                    "params": step.get("params"),
                }
                new_config = copy.copy(self.config)
                self.set_catalog(
                    ResampleLayer(
                        config=new_config, catalog=self.catalog, parameters=step_params
                    ).run()
                )
                self.config.steps.pop()

        img_path_list = [
            self.config.general.input_path / filename
            for filename in self.get_catalog()["filename"]
        ]
        if self.n_jobs == 1:
            image_list = [
                self.process_image(img_path)
                for img_path in tqdm(img_path_list, total=len(img_path_list), desc="Open Images")
            ]
            # image_list = []
            # for _, img_path in tqdm(
            #     enumerate(img_path_list), total=len(img_path_list), desc="Open Images"
            # ):
            #     # image_metadata = self.get_catalog(flag="all").iloc[index].to_dict()
            #     # image_metadata['bit_depth'] = img.dtype.itemsize * 8
            #     img_layer = self.process_image(img_path)
            #     image_list.append(img_layer)

            # image_list = self._stack_images_with_padding(image_list)
        else:
            delayed_image_list = [
                delayed(self.process_image)(img_path)
                for _, img_path in enumerate(img_path_list)
            ]
            # delayed_image_list = []
            # for _, img_path in enumerate(img_path_list):
            #     # image_metadata = self.get_catalog(flag="all").iloc[index].to_dict()
            #     # image_metadata['bit_depth'] = img.dtype.itemsize * 8
            #     delayed_image = delayed(self.process_image)(img_path)
            #     delayed_image_list.append(delayed_image)
            # # Distribute the computation across available cores
            with dask.config.set(scheduler='threads', num_workers=self.n_jobs):
                self.logger.info("Processing images using %s cores", self.n_jobs)
                with ProgressBar():
                    computed_images = compute(*delayed_image_list)
                image_list = list(computed_images)
                # image_list = self._stack_images_with_padding(computed_images)

        # Add the processed images to the step
        self.images.add_step(
            step=self.step_name,
            images=image_list,
            step_metadata=self.step_metadata,
            catalog=self.get_catalog(),
        )
        del image_list
        gc.collect()


    def _pad_to_target_shape(self, array, target_shape, constant_values=0):
        pad_width = [(0, t_dim - a_dim) for a_dim, t_dim in zip(array.shape, target_shape)]
        if self.n_jobs == 1:
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=constant_values)
        else:
            padded_array = da.pad(array, pad_width, mode='constant', constant_values=constant_values)
        return padded_array

    def _stack_images_with_padding(self, image_list):
        constant_values = 0
        target_shape = tuple(max(img.shape[dim] for img in image_list) for dim in range(len(image_list[0].shape)))

        padded_images = [self._pad_to_target_shape(img, target_shape, constant_values) for img in image_list]
        if self.n_jobs == 1:
            stacked_images = np.stack(padded_images, axis=0)
            mask = np.zeros_like(stacked_images, dtype=bool)
        else:
            stacked_images = da.stack(padded_images, axis=0)
            mask = da.zeros_like(stacked_images, dtype=bool)


        for i, img in enumerate(image_list):
            mask[i, :img.shape[0], :img.shape[1]] = False
            mask[i, img.shape[0]:, img.shape[1]:] = True

        if self.n_jobs == 1:
            masked_images = np.ma.masked_array(stacked_images, mask=mask)
        else:
            masked_images = da.ma.masked_array(stacked_images, mask=mask)
        return masked_images

    def process_image(self, img_path):
        """Process a single image"""
        img = self.open_image(img_path=img_path)
        # img_layer = ImageLayer(
        #     image=img,
        #     name=image_metadata.get("filename"),
        #     image_metadata=image_metadata,
        #     step_order=self.images.get_last_step_order(),
        #     step_name=self.step_name,
        # )
        # del img
        gc.collect()
        return img

    def open_image(self, img_path: str) -> Union[np.ndarray, dask.array.core.Array]:
        """ Open an image file

        Args:
            img_path (str): The path to the image file

        Raises:
            ValueError: Failed to open the image

        Returns:
            np.ndarray: The image data
        """
        if self.n_jobs == 1:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # if len(img.shape) == 2:
            #     img = np.expand_dims(img, axis=-1)
        else:
            img = dask_image.imread.imread(img_path)
            img = np.squeeze(img)
            # if len(img.shape) == 2:
            #     img = da.expand_dims(img, axis=-1)
        if img is None:
            if self.raise_error:
                self.logger.error("Failed to open the image: %s", img_path)
                raise ValueError("Failed to open the image")
            self.logger.warning("Failed to open the image: %s", img_path)
            return None
        return img

    def extract_exif(self):
        img_path_list = [
            self.config.general.input_path / filename
            for filename in self.get_catalog()["filename"]
        ]
        exif_list = []
        for img_path in img_path_list:
            exif_list.append(OpenLayer.extract_exif_single(img_path))
        self.set_catalog(
            self.get_catalog(flag="all").merge(
                pd.DataFrame(exif_list), on="filename", how="left"
            )
        )

    @staticmethod
    def extract_exif_single(img_path):
        img_pil = Image.open(img_path)
        exif_data = img_pil.getexif()
        exif = {}
        if exif_data is not None:
            exif["filename"] = img_path.name
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif[tag_name] = value
        return exif
