""" Open raw image file
"""

import numpy as np
import cv2
from tqdm import tqdm

from paidiverpy import Paidiverpy
from paidiverpy.image_layer import ImageLayer
from paidiverpy.convert_layer.params import (
    CONVERT_LAYER_METHODS,
    BitParams,
    ToParams,
    BayerPatternParams,
    NormalizeParams,
    ResizeParams,
    CropParams,
)


class ConvertLayer(Paidiverpy):
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
        step_name=None,
        parameters=None,
        config_index=None,
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
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(
            self.config.steps[self.config_index]
        )

    def run(self, add_new_step=True):
        mode = self.step_metadata.get("mode")
        if not mode:
            raise ValueError("The mode is not defined in the configuration file.")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}

        if mode not in CONVERT_LAYER_METHODS:
            raise ValueError(f"Unsupported mode: {mode}")

        params = CONVERT_LAYER_METHODS[mode]["params"](**params)

        method_info = CONVERT_LAYER_METHODS[mode]
        method_name = method_info["method"]

        images = self.images.get_step(step=len(self.images.images) - 1, by_order=True)
        image_list = []

        for index, img in tqdm(
            enumerate(images), total=len(images), desc="Processing Images"
        ):
            method = getattr(self, method_name)
            img_data = method(img, params=params)
            image_metadata = self.get_catalog(flag="all").iloc[index].to_dict()
            if mode == "convert_bits":
                image_metadata["bit_depth"] = params.output_bits
            if mode == "normalize":
                image_metadata["normalize"] = (params.min, params.max)
            img = ImageLayer(
                image=img_data,
                image_metadata=image_metadata,
                step_order=self.images.get_last_step_order(),
                step_name=self.step_name,
            )
            image_list.append(img)
        if not test:
            self.step_name = (
                f"convert_{self.config_index}" if not self.step_name else self.step_name
            )
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=image_list,
                    step_metadata=self.step_metadata,
                )
            else:
                self.images.images[-1] = image_list
                return self.images

    def convert_bits(self, img, params: BitParams = BitParams()):
        image_data = img.image
        if params.autoscale:
            try:
                result = np.float32(image_data) - np.min(image_data)
                result[result < 0.0] = 0.0
                if np.max(image_data) != 0:
                    result = result / np.max(image_data)
                if params.output_bits == 8:
                    img_bit = np.uint8(255 * result)
                elif params.output_bits == 16:
                    img_bit = np.uint16(65535 * result)
                elif params.output_bits == 32:
                    img_bit = np.float32(result)
            except Exception as e:
                self.logger.error("Failed to autoscale the image: %s", e)
                if self.raise_error:
                    raise ValueError(f"Failed to autoscale the image: {str(e)}") from e
                img_bit = image_data
        else:
            if params.output_bits == 8:
                img_bit = np.uint8(255)
            elif params.output_bits == 16:
                img_bit = np.uint16(65535)
            elif params.output_bits == 32:
                img_bit = np.float32(result)
        return img_bit

    def channel_convert(self, img, params: ToParams = ToParams()):
        image_data = img.image
        try:
            if params.to == "RGB":
                if len(image_data.shape) != 3 and image_data.shape[2] != 3:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
            elif params.to == "gray":
                if params.channel_selector in [0, 1, 2]:
                    image_data = image_data[:, :, params.channel_selector]
                else:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.logger.error(f"Failed to convert the image to {params.to}: {str(e)}")
            if self.raise_error:
                raise ValueError(
                    f"Failed to convert the image to {params.to}: {str(e)}"
                ) from e
        return image_data

    def get_bayer_pattern(self, img, params: BayerPatternParams = BayerPatternParams()):
        # Determine the number of channels in the input image
        image_data = img.image
        if len(image_data.shape) == 3:
            img_channels = image_data.shape[2]
        else:
            img_channels = 1
        if img_channels == 1:
            if params.bayer_pattern == "RG":
                bayer_pattern = cv2.COLOR_BAYER_RG2RGB
            elif params.bayer_pattern == "BG":
                bayer_pattern = cv2.COLOR_BAYER_BG2RGB
            elif params.bayer_pattern == "GR":
                bayer_pattern = cv2.COLOR_BAYER_GR2RGB
            elif params.bayer_pattern == "GB":
                bayer_pattern = cv2.COLOR_BAYER_GB2RGB
            else:
                self.logger.warning(
                    "Invalid Bayer pattern for a single-channel image: %s",
                    params.bayer_pattern,
                )
                if self.raise_error:
                    raise ValueError(
                        "Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'."
                    )
                return image_data
        elif img_channels in [3, 4]:
            self.logger.warning(
                "Unsupported number of channels in the image: %s", img_channels
            )
            if self.raise_error:
                raise ValueError(
                    "Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'."
                )
            return image_data
        else:
            self.logger.warning(
                "Unsupported number of channels in the image: %s", img_channels
            )
            if self.raise_error:
                raise ValueError("Unsupported number of channels in the image.")
            return image_data
        return cv2.cvtColor(image_data, bayer_pattern)

    def normalize_image(self, img, params: NormalizeParams = NormalizeParams()):
        image_data = img.image
        try:
            return cv2.normalize(
                image_data, image_data, params.min, params.max, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        except Exception as e:
            self.logger.error(f"Failed to normalize the image: {str(e)}")
            if self.raise_error:
                raise ValueError(f"Failed to normalize the image: {str(e)}") from e
        return image_data

    def resize(self, img, params: ResizeParams = ResizeParams()):
        image_data = img.image
        try:
            return cv2.resize(
                image_data, (params.min, params.max), interpolation=cv2.INTER_LANCZOS4
            )
        except Exception as e:
            self.logger.error(f"Failed to resize the image: {str(e)}")
            if self.raise_error:
                raise ValueError(f"Failed to resize the image: {str(e)}") from e
        return image_data

    def crop_images(self, img, params: CropParams = CropParams()):
        image_data = img.image
        try:
            start_x, end_x = params.x[0]
            start_y, end_y = params.y[1]
            if (
                start_x < 0
                or end_x > image_data.shape[0]
                or start_y < 0
                or end_y > image_data.shape[1]
            ):
                raise ValueError("Crop range is out of bounds.")
            return image_data[start_x:end_x, start_y:end_y, :]
        except Exception as e:
            self.logger.error("Failed to crop the image: %s", str(e))
            if self.raise_error:
                raise ValueError(f"Failed to crop the image: {str(e)}") from e
        return image_data
