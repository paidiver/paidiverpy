"""Convert Layer.

Convert the images in the convert layer based on the configuration file or
parameters.
"""

import logging
import cv2
import dask
import dask.array as da
import dask.delayed
import numpy as np
from dask import compute
from dask.diagnostics import ProgressBar
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.convert_params import BayerPatternParams
from paidiverpy.config.convert_params import BitParams
from paidiverpy.config.convert_params import CropParams
from paidiverpy.config.convert_params import NormalizeParams
from paidiverpy.config.convert_params import ResizeParams
from paidiverpy.config.convert_params import ToParams
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from utils import DynamicConfig
from utils import raise_value_error

EIGHT_BITS = 8
SIXTEEN_BITS = 16
THIRTY_TWO_BITS = 32

class ConvertLayer(Paidiverpy):
    """Process the images in the convert layer.

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
        config_index (int): The index of the configuration.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
        track_changes (bool): Whether to track changes. Defaults to True.
        n_jobs (int): The number of jobs to run in parallel.
    """

    def __init__(
        self,
        config_file_path: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        metadata_path: str | None = None,
        metadata_type: str | None = None,
        metadata: MetadataParser = None,
        config: Configuration = None,
        logger: logging.Logger | None = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        parameters: dict | None = None,
        config_index: int | None = None,
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
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])

    def run(self, add_new_step: bool = True) -> ImagesLayer | None:
        """Run the convert layer steps on the images based on the configuration.

        Run the convert layer steps on the images based on the configuration
        file or parameters.

        Args:
            add_new_step (bool, optional): Whether to add a new step to the images object.
        Defaults to True.

        Raises:
            ValueError: The mode is not defined in the configuration file.

        Returns:
            Union[ImagesLayer, None]: The images object with the new step added.
        """
        mode = self.step_metadata.get("mode")
        if not mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, CONVERT_LAYER_METHODS, mode)
        images = self.images.get_step(step=len(self.images.images) - 1, by_order=True)
        if self.n_jobs == 1:
            image_list = self.process_sequentially(images, method, params)
        else:
            image_list = self.process_parallel(images, method, params)
        if not test:
            self.step_name = f"convert_{self.config_index}" if not self.step_name else self.step_name
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=image_list,
                    step_metadata=self.step_metadata,
                    metadata=self.get_metadata(),
                    track_changes=self.track_changes,
                )
                return None
            self.images.images[-1] = image_list
            return self.images
        return None

    def process_sequentially(self, images: list[np.ndarray], method: callable, params: dict) -> list[np.ndarray]:
        """Process the images sequentially.

        Args:
            images (List[np.ndarray]): The list of images to process.
            method (callable): The method to apply to the images.
            params (dict): The parameters for the method.

        Returns:
            List[np.ndarray]: The list of processed images.
        """
        return [method(img, params=params) for img in images]

    def process_parallel(
        self, images: list[da.core.Array], method: callable, params: DynamicConfig,
    ) -> list[np.ndarray]:
        """Process the images in parallel.

        Args:
            images (List[da.core.Array]): The list of images to process.
            method (callable): The method to apply to the images.
            params (DynamicConfig): The parameters for the method.

        Returns:
            List[da.core.Array]: The list of processed images.
        """
        delayed_images = [dask.delayed(method)(img, params) for img in images]
        with dask.config.set(scheduler="threads", num_workers=self.n_jobs):
            self.logger.info("Processing images using %s cores", self.n_jobs)
            with ProgressBar():
                delayed_images = compute(*delayed_images)
        return [da.from_array(img) for img in delayed_images]

    def convert_bits(self, image_data: np.ndarray, params: BitParams = None) -> np.ndarray:
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
        if params.output_bits == EIGHT_BITS:
            image_data = np.uint8(image_data * 255)
        elif params.output_bits == SIXTEEN_BITS:
            image_data = np.uint16(image_data * 65535)
        elif params.output_bits == THIRTY_TWO_BITS:
            image_data = np.float32(image_data)
        else:
            self.logger.warning("Unsupported output bits: %s", params.output_bits)
            if self.raise_error:
                msg = f"Unsupported output bits: {params.output_bits}"
                raise ValueError(msg)

        return image_data

    def channel_convert(self, image_data: np.ndarray, params: ToParams = None) -> np.ndarray:
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
        except Exception as e:
            self.logger.warning("Failed to convert the image to %s: %s", params.to, e)
            if self.raise_error:
                msg = f"Failed to convert the image to {params.to}: {e}"
                raise_value_error(msg)
        return image_data

    def get_bayer_pattern(
        self, image_data: np.ndarray, params: BayerPatternParams = None,
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
            self.logger.warning(
                "Invalid Bayer pattern for a single-channel image: %s",
                params.bayer_pattern,
            )
            if self.raise_error:
                msg = "Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'."
                raise ValueError(
                    msg,
                )
            return image_data
        try:
            bayer_pattern = {
                "RG": cv2.COLOR_BAYER_RG2RGB,
                "BG": cv2.COLOR_BAYER_BG2RGB,
                "GR": cv2.COLOR_BAYER_GR2RGB,
                "GB": cv2.COLOR_BAYER_GB2RGB,
            }[params.bayer_pattern]
        except KeyError as exc:
            self.logger.warning(
                "Invalid Bayer pattern for a single-channel image: %s",
                params.bayer_pattern,
            )
            if self.raise_error:
                msg = "Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'."
                raise KeyError(
                    msg,
                ) from exc

            return image_data
        return cv2.cvtColor(image_data, bayer_pattern)


    def normalize_image(self, image_data: np.ndarray, params: NormalizeParams = None) -> np.ndarray:
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
            return cv2.normalize(
                image_data,
                image_data,
                params.min,
                params.max,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        except Exception as e:
            self.logger.warning("Failed to normalize the image: %s", e)
            if self.raise_error:
                msg = f"Failed to normalize the image: {e!s}"
                raise ValueError(msg) from e
        return image_data

    def resize(self, image_data: np.ndarray, params: ResizeParams = None) -> np.ndarray:
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
        except Exception as e:
            self.logger.warning("Failed to resize the image: %s", e)
            if self.raise_error:
                msg = f"Failed to resize the image: {e!s}"
                raise ValueError(msg) from e
        return image_data

    def crop_images(self, image_data: np.ndarray, params: CropParams = None) -> np.ndarray:
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
            start_x, end_x = params.x[0]
            start_y, end_y = params.y[1]
            if start_x < 0 or end_x > image_data.shape[1] or start_y < 0 or end_y > image_data.shape[2]:
                msg = "Crop range is out of bounds."
                raise_value_error(msg)
            return image_data[:, start_x:end_x, start_y:end_y, :]
        except Exception as e:
            self.logger.warning("Failed to crop the image: %s", e)
            if self.raise_error:
                msg = f"Failed to crop the image: {e!s}"
                raise ValueError(msg) from e
        return image_data
