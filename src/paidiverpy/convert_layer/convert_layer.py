"""Convert Layer.

Convert the images in the convert layer based on the configuration file or
parameters.
"""

import logging
import cv2
import numpy as np
from dask.distributed import Client
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.models.convert_params import BitParams
from paidiverpy.models.convert_params import CropParams
from paidiverpy.models.convert_params import NormalizeParams
from paidiverpy.models.convert_params import ResizeParams
from paidiverpy.models.convert_params import ToParams
from paidiverpy.utils.data import EIGHT_BITS
from paidiverpy.utils.data import EIGHT_BITS_SIZE
from paidiverpy.utils.data import NUM_CHANNELS_GREY
from paidiverpy.utils.data import NUM_CHANNELS_RGB
from paidiverpy.utils.data import NUM_CHANNELS_RGBA
from paidiverpy.utils.data import SIXTEEN_BITS
from paidiverpy.utils.data import SIXTEEN_BITS_SIZE
from paidiverpy.utils.data import THIRTY_TWO_BITS
from paidiverpy.utils.data import THIRTY_TWO_BITS_SIZE
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.logging_functions import check_raise_error


class ConvertLayer(Paidiverpy):
    """Process the images in the convert layer.

    This class provides various methods to convert images according to specified
    configurations, such as resizing, normalizing, bit depth conversion, and channel conversion.

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
        config_index (int): The index of the configuration.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        parameters: dict,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        client: Client | None = None,
        config_index: int | None = None,
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
        self.config_index = self.config.add_step(config_index, parameters, step_class=ConvertLayer)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.raise_error = self._calculate_raise_error()
        self.layer_methods = CONVERT_LAYER_METHODS

    @staticmethod
    def convert_bits(image_data: np.ndarray, metadata: dict | None = None, params: BitParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Convert the image to the specified number of bits.

        Args:
            image_data (np.ndarray): The image data.
            metadata (dict, optional): The metadata for the image.
            params (BitParams, optional): The parameters for the bit conversion.
            **kwargs: Additional keyword arguments.

        Defaults to BitParams().

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, BitParams, **kwargs)
        try:
            bit = metadata.get("bit_depth", image_data.dtype.itemsize)

            if params.output_bits == EIGHT_BITS and bit != EIGHT_BITS_SIZE:
                image_data, metadata = ConvertLayer.normalize_image(image_data, metadata)
                image_data = np.uint8(image_data * 255)
            elif params.output_bits == SIXTEEN_BITS and bit != SIXTEEN_BITS_SIZE:
                image_data, metadata = ConvertLayer.normalize_image(image_data, metadata)
                image_data = np.uint16(image_data * 65535)
            elif params.output_bits == THIRTY_TWO_BITS and bit != THIRTY_TWO_BITS_SIZE:
                image_data, metadata = ConvertLayer.normalize_image(image_data, metadata)
                image_data = np.float32(image_data)
            else:
                msg = f"Unsupported output bits or image already within provided format: {params.output_bits}"
                raise_value_error(msg)
            metadata["bit_depth"] = params.output_bits / 8
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to convert the image to {params.output_bits} bits: {e!s}"
            check_raise_error(params.raise_error, msg)

        return image_data, metadata

    @staticmethod
    def channel_convert(image_data: np.ndarray, metadata: dict | None = None, params: ToParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Convert the image to the specified channel.

        Args:
            image_data (np.ndarray): The image data.
            metadata (dict, optional): The metadata for the image.
            params (ToParams, optional): The parameters for the channel conversion.
                Defaults to ToParams().
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: The image is already in RGB format.
            ValueError: The image is already in grayscale.
            ValueError: Failed to convert the image to {params.to}: {str(e)}

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, ToParams, **kwargs)
        num_channels = metadata.get("num_channels", image_data.shape[-1])

        conversion_map = {
            "RGB": {
                NUM_CHANNELS_RGBA: cv2.COLOR_RGBA2RGB,
                NUM_CHANNELS_GREY: cv2.COLOR_GRAY2RGB,
                NUM_CHANNELS_RGB: None,
            },
            "RGBA": {
                NUM_CHANNELS_RGB: cv2.COLOR_RGB2RGBA,
                NUM_CHANNELS_GREY: cv2.COLOR_GRAY2RGBA,
                NUM_CHANNELS_RGBA: None,
            },
            "gray": {
                NUM_CHANNELS_RGBA: cv2.COLOR_RGBA2GRAY,
                NUM_CHANNELS_RGB: cv2.COLOR_RGB2GRAY,
                NUM_CHANNELS_GREY: None,
            },
        }
        try:
            conversion = conversion_map.get(params.to, {}).get(num_channels)
            if conversion is None:
                raise_value_error(f"The image is already in {params.to.upper()} format.")
            image_data = cv2.cvtColor(image_data, conversion)
            metadata["num_channels"] = 3 if params.to == "RGB" else 4 if params.to == "RGBA" else 1
            if params.to == "gray":
                image_data = np.expand_dims(image_data, axis=-1)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to convert the image to {params.to}: {e!s}"
            check_raise_error(params.raise_error, msg)

        return image_data, metadata

    @staticmethod
    def normalize_image(
        image_data: np.ndarray, metadata: dict | None = None, params: NormalizeParams = None, **kwargs: dict
    ) -> tuple[np.ndarray, dict]:
        """Normalize the image data.

        Args:
            image_data (np.ndarray): The image data.
            metadata (dict, optional): The metadata for the image.
            params (NormalizeParams, optional): The parameters for the image normalization.
                Defaults to NormalizeParams().
            **kwargs: Additional keyword arguments.

        Defaults to NormalizeParams().

        Raises:
            ValueError: Failed to normalize the image: {str(e)}

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, NormalizeParams, **kwargs)
        try:
            if params.method == "minmax":
                if params.min > params.max:
                    msg = "The minimum value must be less than the maximum value."
                    raise_value_error(msg)
                normalized_image = cv2.normalize(
                    image_data.astype(np.float32),
                    None,
                    params.min,
                    params.max,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F,
                )
                image_data = np.clip(normalized_image, params.min, params.max)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to normalize the image: {e!s}"
            check_raise_error(params.raise_error, msg)

        return image_data, metadata

    @staticmethod
    def resize(image_data: np.ndarray, metadata: dict | None = None, params: ResizeParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Resize the image data.

        Args:
            image_data (np.ndarray): The image data.
            metadata (dict, optional): The metadata for the image.
            params (ResizeParams, optional): The parameters for the image resizing.
                Defaults to ResizeParams().
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: Failed to resize the image: {str(e)}

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, ResizeParams, **kwargs)
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        try:
            if params.size is not None:
                if params.preserve_aspect:
                    heigth, width = image_data.shape[:2]
                    target_w = params.size["width"]
                    target_h = params.size["height"]
                    scale = min(target_w / width, target_h / heigth)
                    new_w, new_h = int(width * scale), int(heigth * scale)
                else:
                    new_w = params.size["width"]
                    new_h = params.size["height"]
            elif params.scale is not None:
                if isinstance(params.scale, int | float):
                    fx = fy = params.scale
                else:
                    fx = params.scale["x"]
                    fy = params.scale["y"]
                new_w = int(image_data.shape[1] * fx)
                new_h = int(image_data.shape[0] * fy)
            image_data = cv2.resize(image_data, (new_w, new_h), interpolation=interp_map[params.interpolation])
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to resize the image: {e!s}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def crop_images(image_data: np.ndarray, metadata: dict | None = None, params: CropParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Crop the image data.

        Args:
            image_data (np.ndarray): The image data.
            metadata (dict, optional): The metadata for the image.
            params (CropParams, optional): The parameters for the image cropping.
                Defaults to CropParams().
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: The crop size is larger than the image size.
            ValueError: top_left must be provided when mode='topleft'.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, CropParams, **kwargs)
        try:
            height, width = image_data.shape[:2]
            crop_h, crop_w = ConvertLayer._get_crop_size(height, width, params.size, params.size_type)
            if params.mode == "center":
                y1 = max((height - crop_h) // 2, 0)
                x1 = max((width - crop_w) // 2, 0)
            elif params.mode == "random":
                rng = np.random.default_rng()
                x1 = rng.integers(0, max(width - crop_w + 1, 1))
                y1 = rng.integers(0, max(height - crop_h + 1, 1))
            elif params.mode == "top_left":
                y1 = params.top_left["top"]
                x1 = params.top_left["left"]
            y2, x2 = y1 + crop_h, x1 + crop_w
            pad_top = max(-y1, 0)
            pad_left = max(-x1, 0)
            pad_bottom = max(y2 - height, 0)
            pad_right = max(x2 - width, 0)

            if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
                pad_width = (
                    (pad_top, pad_bottom),
                    (pad_left, pad_right),
                )
                pad_width += ((0, 0),)
                image_data = np.pad(image_data, pad_width, mode="reflect")
                y1 += pad_top
                x1 += pad_left
                y2 = y1 + crop_h
                x2 = x1 + crop_w
            image_data = image_data[y1:y2, x1:x2]
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to crop the image: {e!s}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def _get_crop_size(height: int, width: int, size: float | list, size_type: str) -> tuple:
        """Get the crop size.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.
            size (tuple): The size of the crop.
            size_type (str): The type of the crop.

        Returns:
            tuple: The height and width of the crop.
        """
        if isinstance(size, int | float):
            crop_h = crop_w = size
        elif isinstance(size, dict):
            crop_h = size["height"]
            crop_w = size["width"]
        else:
            msg = "Size must be a float or dict"
            raise_value_error(msg)
        if size_type == "percent":
            crop_h = int(height * crop_h)
            crop_w = int(width * crop_w)
        if crop_h > height or crop_w > width:
            msg = "Crop size is larger than the image size."
            raise_value_error(msg)
        return crop_h, crop_w
