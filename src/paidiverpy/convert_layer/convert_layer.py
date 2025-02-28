"""Convert Layer.

Convert the images in the convert layer based on the configuration file or
parameters.
"""

import logging
import cv2
import numpy as np
from dask.distributed import Client
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.convert_params import BayerPatternParams
from paidiverpy.config.convert_params import BitParams
from paidiverpy.config.convert_params import CropParams
from paidiverpy.config.convert_params import NormalizeParams
from paidiverpy.config.convert_params import ResizeParams
from paidiverpy.config.convert_params import ToParams
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils.data import EIGHT_BITS
from paidiverpy.utils.data import EIGHT_BITS_SIZE
from paidiverpy.utils.data import NUM_CHANNELS_GREY
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
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        parameters: dict | None = None,
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
        if parameters:
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.layer_methods = CONVERT_LAYER_METHODS

    @staticmethod
    def convert_bits(image_data: np.ndarray, params: BitParams = None) -> np.ndarray:
        """Convert the image to the specified number of bits.

        Args:
            image_data (np.ndarray): The image data.
            params (BitParams, optional): The parameters for the bit conversion.
        Defaults to BitParams().

        Returns:
            np.ndarray: The image data with the specified number of bits.
        """
        if params is None:
            params = BitParams()

        bit = image_data.dtype.itemsize

        if params.output_bits == EIGHT_BITS and bit != EIGHT_BITS_SIZE:
            image_data = ConvertLayer.normalize_image(image_data)
            image_data = np.uint8(image_data * 255)
        elif params.output_bits == SIXTEEN_BITS and bit != SIXTEEN_BITS_SIZE:
            image_data = ConvertLayer.normalize_image(image_data)
            image_data = np.uint16(image_data * 65535)
        elif params.output_bits == THIRTY_TWO_BITS and bit != THIRTY_TWO_BITS_SIZE:
            image_data = ConvertLayer.normalize_image(image_data)
            image_data = np.float32(image_data)
        else:
            msg = f"Unsupported output bits or image already within provided format: {params.output_bits}"
            check_raise_error(params.raise_error, msg)

        return image_data

    @staticmethod
    def channel_convert(image_data: np.ndarray, params: ToParams = None) -> np.ndarray:
        """Convert the image to the specified channel.

        Args:
            image_data (np.ndarray): The image data.
            params (ToParams, optional): The parameters for the channel conversion.
        Defaults to ToParams().

        Raises:
            ValueError: The image is already in RGB format.
            ValueError: The image is already in grayscale.
            ValueError: Failed to convert the image to {params.to}: {str(e)}

        Returns:
            np.ndarray: The image data with the specified channel.
        """
        if params is None:
            params = ToParams()
        try:
            if params.to == "RGB":
                if image_data.shape[-1] == 1:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
                else:
                    raise_value_error("The image is already in RGB format.")
            elif params.to == "gray":
                if image_data.shape[-1] == 1:
                    raise_value_error("The image is already in grayscale.")
                if params.channel_selector in [0, 1, 2]:
                    image_data = image_data[:, :, params.channel_selector]
                else:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to convert the image to {params.to}: {e!s}"
            check_raise_error(params.raise_error, msg)
        return image_data

    @staticmethod
    def get_bayer_pattern(
        image_data: np.ndarray,
        params: BayerPatternParams = None,
    ) -> np.ndarray:
        """Convert the image to the specified Bayer pattern.

        Args:
            image_data (np.ndarray): The image data.
            params (BayerPatternParams, optional): The parameters for the Bayer pattern conversion.
        Defaults to BayerPatternParams().

        Raises:
            ValueError: Invalid Bayer pattern for a single-channel image.
            KeyError: Invalid Bayer pattern for a single-channel image.
        Expected 'RG', 'BG', 'GR', or 'GB'.

        Returns:
            np.ndarray: The image data with the specified Bayer pattern.
        """
        if params is None:
            params = BayerPatternParams()
        if image_data.shape[-1] != 1:
            msg = "Invalid Bayer pattern for a single-channel image."
            check_raise_error(params.raise_error, msg)
            return image_data
        try:
            bayer_pattern = {
                "RG": cv2.COLOR_BAYER_RG2RGB,
                "BG": cv2.COLOR_BAYER_BG2RGB,
                "GR": cv2.COLOR_BAYER_GR2RGB,
                "GB": cv2.COLOR_BAYER_GB2RGB,
            }[params.bayer_pattern]
        except KeyError:
            msg = "Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'."
            check_raise_error(params.raise_error, msg)
            return image_data
        return cv2.cvtColor(image_data, bayer_pattern)

    @staticmethod
    def normalize_image(image_data: np.ndarray, params: NormalizeParams = None) -> np.ndarray:
        """Normalize the image data.

        Args:
            image_data (np.ndarray): The image data.
            params (NormalizeParams, optional): The parameters for the image normalization.
        Defaults to NormalizeParams().

        Raises:
            ValueError: Failed to normalize the image: {str(e)}

        Returns:
            np.ndarray: The normalized image data.
        """
        if params is None:
            params = NormalizeParams()
        try:
            normalized_image = cv2.normalize(
                image_data.astype(np.float32),
                None,
                params.min,
                params.max,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            return np.clip(normalized_image, params.min, params.max)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to normalize the image: {e!s}"
            check_raise_error(params.raise_error, msg)
        return image_data

    @staticmethod
    def resize(image_data: np.ndarray, params: ResizeParams = None) -> np.ndarray:
        """Resize the image data.

        Args:
            image_data (np.ndarray): The image data.
            params (ResizeParams, optional): The parameters for the image resizing.
        Defaults to ResizeParams().

        Raises:
            ValueError: Failed to resize the image: {str(e)}

        Returns:
            np.ndarray: The resized image data.
        """
        if params is None:
            params = ResizeParams()
        try:
            return cv2.resize(image_data, (params.min, params.max), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to resize the image: {e!s}"
            check_raise_error(params.raise_error, msg)
        return image_data

    @staticmethod
    def crop_images(image_data: np.ndarray, params: CropParams = None) -> np.ndarray:
        """Crop the image data.

        Args:
            image_data (np.ndarray): The image data.
            params (CropParams, optional): The parameters for the image cropping.
        Defaults to CropParams().

        Raises:
            ValueError: Crop range is out of bounds.
            ValueError: Failed to crop the image: {str(e)}

        Returns:
            np.ndarray: The cropped image data.
        """
        if params is None:
            params = CropParams()
        try:
            start_x, end_x = params.x
            start_y, end_y = params.y
            if start_x < 0 or end_x > image_data.shape[0] or start_y < 0 or end_y > image_data.shape[1]:
                msg = "Crop range is out of bounds."
                raise_value_error(msg)
            if len(image_data.shape) == NUM_CHANNELS_GREY:
                return image_data[start_y:end_y, start_x:end_x]
            return image_data[start_y:end_y, start_x:end_x:]
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to crop the image: {e!s}"
            check_raise_error(params.raise_error, msg)
        return image_data
