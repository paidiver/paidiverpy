""" Open raw image file
"""

import gc
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import cv2
from tqdm import tqdm
from paidiverpy import Paidiverpy
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.image_layer import ImageLayer
from paidiverpy.resample_layer import ResampleLayer
import copy


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
        if self.step_metadata.get("sampling"):
            for step in self.step_metadata.get("sampling"):
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

        image_list = []

        for index, img_path in tqdm(
            enumerate(img_path_list), total=len(img_path_list), desc="Processing Images"
        ):
            img = self.open_image(
                img_path=img_path,
            )
<<<<<<< HEAD
            image_metadata = self.get_catalog(flag="all").iloc[index].to_dict()
            image_metadata['bit_depth'] = img.dtype.itemsize * 8
            img = ImageLayer(
                image=img,
                image_metadata=image_metadata,
=======
            img = ImageLayer(
                image=img,
                image_metadata=self.get_catalog(flag="all").iloc[index].to_dict(),
>>>>>>> 0b6637d5876468601d52a016d92742984755764b
                step_order=self.images.get_last_step_order(),
                step_name=self.step_name,
            )
            image_list.append(img)
            del img
            gc.collect()
        self.images.add_step(
            step=self.step_name, images=image_list, step_metadata=self.step_metadata
        )
        del image_list
        gc.collect()

    def open_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
