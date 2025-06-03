"""Colour layer module.

This module contains the ColourLayer class for processing the images in the
colour layer.
"""

import logging
import cv2
import numpy as np
from scipy import ndimage
from skimage import color
from skimage import measure
from skimage import morphology
from skimage import restoration
from skimage.exposure import adjust_gamma
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.filters import scharr
from skimage.filters import unsharp_mask
from skimage.restoration import rolling_ball
from skimage.restoration import wiener
from skimage.segmentation import checkerboard_level_set
from skimage.segmentation import morphological_chan_vese
from skimage.transform import resize
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.models.colour_params import ColourAlterationParams
from paidiverpy.models.colour_params import ContrastAdjustmentParams
from paidiverpy.models.colour_params import DeblurParams
from paidiverpy.models.colour_params import EdgeDetectionParams
from paidiverpy.models.colour_params import GaussianBlurParams
from paidiverpy.models.colour_params import GrayScaleParams
from paidiverpy.models.colour_params import IlluminationCorrectionParams
from paidiverpy.models.colour_params import SharpenParams
from paidiverpy.utils.data import DEFAULT_BITS
from paidiverpy.utils.data import NUM_CHANNELS_GREY
from paidiverpy.utils.data import NUM_CHANNELS_RGB
from paidiverpy.utils.data import NUM_CHANNELS_RGBA
from paidiverpy.utils.data import NUM_DIMENSIONS
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.logging_functions import check_raise_error


class ColourLayer(Paidiverpy):
    """ColourLayer class.

    This class contains the methods for processing the images in the colour layer.

    Args:
        parameters (dict): The parameters for the step.
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
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )

        self.step_name = step_name
        self.config_index = self.config.add_step(config_index, parameters, step_class=ColourLayer)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.raise_error = self._calculate_raise_error()
        self.layer_methods = COLOUR_LAYER_METHODS

    @staticmethod
    def grayscale(image_data: np.ndarray, metadata: dict | None = None, params: GrayScaleParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Convert the image to grayscale.

        Method to convert the image to grayscale.

        Args:
            image_data (np.ndarray): The input image.
            metadata (dict, optional): Metadata for the image.
            params (GrayScaleParams, optional): Parameters for the grayscale conversion.
                Defaults to GrayScaleParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: If the input image does not have 3 channels or 4 channels with alpha.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, GrayScaleParams, **kwargs)

        num_channels = metadata.get("num_channels", image_data.shape[-1])

        try:
            if num_channels not in (NUM_CHANNELS_RGB, NUM_CHANNELS_RGBA):
                msg = "Input image must have 3 or 4 channels in the last dimension."
                raise_value_error(msg)
            image_data = ColourLayer._apply_grayscale_conversion(image_data, params)
            image_data = np.expand_dims(image_data, axis=-1)
            if params.invert_colours:
                image_data = 255 - image_data
        except Exception as e:  # noqa: BLE001
            msg = f"Error converting image to grayscale: {e}"
            check_raise_error(params.raise_error, msg)
        metadata["num_channels"] = 1

        return image_data, metadata

    @staticmethod
    def _apply_grayscale_conversion(image_data: np.ndarray, params: GrayScaleParams) -> np.ndarray:
        """GrayScale conversion.

        Apply the grayscale conversion method specified by params

        Args:
            image_data (np.ndarray): The input image.
            params (GrayScaleParams): Parameters for the grayscale conversion.

        Returns:
            np.ndarray: The grayscale image.
        """
        if params.method == "average":
            return np.mean(image_data, axis=-1).astype(image_data.dtype)
        if params.method == "luminosity":
            band1 = 0.2126 * image_data[..., 0]
            band2 = 0.7152 * image_data[..., 1]
            band3 = 0.0722 * image_data[..., 2]
            return (band1 + band2 + band3).astype(image_data.dtype)
        if params.method == "desaturation":
            return ((np.max(image_data, axis=-1) + np.min(image_data, axis=-1)) / 2).astype(image_data.dtype)
        return cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def gaussian_blur(
        image_data: np.ndarray, metadata: dict | None = None, params: GaussianBlurParams = None, **kwargs: dict
    ) -> tuple[np.ndarray, dict]:
        """Gaussian blur.

        Method to apply Gaussian blur to the image.

        Args:
            image_data (np.ndarray): The image to apply Gaussian blur.
            metadata (dict, optional): Metadata for the image.
            params (GaussianBlurParams, optional): the parameters for the method.
                Defaults to GaussianBlurParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: Error applying Gaussian blur.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, GaussianBlurParams, **kwargs)
        try:
            image_data = cv2.GaussianBlur(image_data, (0, 0), params.sigma)
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying Gaussian blur: {e}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def sharpen(image_data: np.ndarray, metadata: dict | None = None, params: SharpenParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Sharpening.

        Method to apply sharpening to the image.

        Args:
            image_data (np.ndarray): The image to apply sharpening.
            metadata (dict, optional): Metadata for the image.
            params (SharpenParams, optional): Params for method. Defaults to SharpenParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: Error applying sharpening.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, SharpenParams, **kwargs)

        try:
            bits = image_data.dtype.itemsize * DEFAULT_BITS
            image_data = unsharp_mask(image_data, radius=params.alpha, amount=params.beta)
            multiply_factor = 255 if bits == DEFAULT_BITS else 65535
            image_data = np.clip(image_data * multiply_factor, 0, multiply_factor).astype(
                np.uint8 if bits == DEFAULT_BITS else np.uint16,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying sharpening: {e}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def contrast_adjustment(
        image_data: np.ndarray,
        metadata: dict | None = None,
        params: ContrastAdjustmentParams = None,
        **kwargs: dict,
    ) -> tuple[np.ndarray, dict]:
        """Contrast adjustment.

        Method to apply contrast adjustment to the image.

        Args:
            image_data (np.ndarray): The image to apply contrast adjustment.
            metadata (dict, optional): Metadata for the image.
            params (ContrastAdjustmentParams, optional): Params for method.
                Defaults to ContrastAdjustmentParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: Error applying contrast adjustment.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, ContrastAdjustmentParams, **kwargs)

        try:
            method = params.method
            kernel_size = params.kernel_size
            if params.kernel_size:
                if isinstance(params.kernel_size, dict):
                    kernel_size = list(params.kernel_size.values())
                else:
                    kernel_size = [params.kernel_size for _ in range(image_data.ndim)]
            clip_limit = params.clip_limit
            gamma_value = params.gamma_value
            bits = image_data.dtype.itemsize * DEFAULT_BITS
            if method == "clahe":
                image_data = equalize_adapthist(image_data, clip_limit=clip_limit, kernel_size=kernel_size)
            elif method == "gamma":
                image_data = adjust_gamma(image_data, gamma=gamma_value)
            multiply_factor = 255 if bits == DEFAULT_BITS else 65535
            image_data = np.clip(image_data * multiply_factor, 0, multiply_factor).astype(
                np.uint8 if bits == DEFAULT_BITS else np.uint16,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying contrast adjustment: {e}"
            check_raise_error(params.raise_error, msg)

        return image_data, metadata

    @staticmethod
    def illumination_correction(
        image_data: np.ndarray,
        metadata: dict | None = None,
        params: IlluminationCorrectionParams = None,
        **kwargs: dict,
    ) -> tuple[np.ndarray, dict]:
        """Illumination correction.

        Method to apply illumination correction to the image.

        Args:
            image_data (np.ndarray): The image to apply illumination correction.
            metadata (dict, optional): Metadata for the image.
            params (IlluminationCorrectionParams, optional): Params for method.
                Defaults to IlluminationCorrectionParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: Error applying illumination correction.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, IlluminationCorrectionParams, **kwargs)

        try:
            method = params.method
            radius = params.radius

            if method == "rolling":
                background = rolling_ball(image_data, radius=radius)
                image_data = image_data - background
            else:
                msg = "Unknown method type. Please use 'rolling'."
                raise_value_error(msg)
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying illumination correction: {e}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def deblur(image_data: np.ndarray, metadata: dict | None = None, params: DeblurParams = None, **kwargs: dict) -> tuple[np.ndarray, dict]:
        """Deblurring.

        Method to apply deblurring to the image.

        Args:
            image_data (np.ndarray): The image to apply deblurring.
            metadata (dict, optional): Metadata for the image.
            params (DeblurParams, optional): Params for method.
                Defaults to DeblurParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: Unknown PSF type. Please use 'gaussian' or 'motion'.
            ValueError: Unknown method type. Please use 'wiener'.
            NotImplementedError: Unknown method type. Please use 'wiener'.
            ValueError: Error applying contrast adjustment.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, DeblurParams, **kwargs)
        num_channels = metadata.get("num_channels", image_data.shape[-1])
        try:
            method = params.method
            psf_type = params.psf_type
            sigma = params.sigma
            angle = params.angle
            if len(image_data.shape) == NUM_DIMENSIONS and num_channels == NUM_CHANNELS_GREY:
                image_data = np.squeeze(image_data)
            if method == "wiener":
                if psf_type == "gaussian":
                    psf = ColourLayer.gaussian_psf(size=image_data.shape, sigma=sigma)
                elif psf_type == "motion":
                    psf = ColourLayer.motion_psf(size=image_data.shape, length=sigma, angle_xy=angle)
                bits = image_data.dtype.itemsize * DEFAULT_BITS
                image_data = np.squeeze(image_data) if num_channels == 1 else image_data
                image_data = wiener(image_data, psf, balance=0.1)
                multiply_factor = 255 if bits == DEFAULT_BITS else 65535
                image_data = np.clip(image_data * multiply_factor, 0, multiply_factor).astype(
                    np.uint8 if bits == DEFAULT_BITS else np.uint16,
                )
            else:
                msg = "Unknown method type. Please use 'wiener'."
                raise_value_error(msg)
            if len(image_data.shape) == NUM_DIMENSIONS_GREY:
                image_data = np.expand_dims(image_data, axis=-1)
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying deblurring: {e}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def colour_alteration(
        image_data: np.ndarray, metadata: dict | None = None, params: ColourAlterationParams = None, **kwargs: dict
    ) -> tuple[np.ndarray, dict]:
        """Apply colour alteration to the image.

        Args:
            image_data (np.ndarray): The image to alter colour channel.
            metadata (dict, optional): Metadata for the image.
            params (ColourAlterationParams, optional): Params for method. Defaults to None.
            **kwargs (dict): Additional keyword arguments.

        Raises:
            ValueError: Unknown method type. Please use 'white_balance'.
            ValueError: Image is gray-scale'.
            e: Error applying colour alteration.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, ColourAlterationParams, **kwargs)
        num_channels = metadata.get("num_channels", image_data.shape[-1])
        try:
            method = params.method
            if method == "white_balance":
                if num_channels == NUM_CHANNELS_GREY:
                    msg = "Image is gray-scale. Cannot apply white balance."
                    raise_value_error(msg)
                image_data = ColourLayer._white_balance(image_data, num_channels)
            else:
                msg = "Unknown method type. Please use 'white_balance'."
                raise_value_error(msg)
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying colour alteration: {e}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def edge_detection(
        image_data: np.ndarray,
        metadata: dict | None = None,
        params: EdgeDetectionParams = None,
        **kwargs: dict,
    ) -> tuple[np.ndarray, dict]:
        """Edge detection.

        Method to apply edge detection to the image.

        Args:
            image_data (np.ndarray): The image to apply edge detection.
            metadata (dict, optional): Metadata for the image.
            params (EdgeDetectionParams, optional): Params for method.
                Defaults to EdgeDetectionParams().
            **kwargs (dict): Additional keyword arguments.

        Raises:
            e: Error applying edge detection.

        Returns:
            tuple[np.ndarray, dict]: The updated image and the updated metadata.
        """
        image_data, metadata, params, _ = Paidiverpy.prepare_inputs(image_data, metadata, params, EdgeDetectionParams, **kwargs)
        num_channels = metadata.get("num_channels", image_data.shape[-1])
        try:
            if params.method == "sobel":
                if num_channels == NUM_CHANNELS_GREY:
                    image_data = np.squeeze(image_data)
                image_data = np.squeeze(image_data) if num_channels == 1 else image_data
                image_data = cv2.Sobel(image_data, cv2.CV_64F, 1, 1, ksize=5)
                return np.expand_dims(image_data, axis=-1), metadata

            if num_channels == NUM_CHANNELS_RGB:
                gray_image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            elif num_channels == NUM_CHANNELS_RGBA:
                gray_image_data = cv2.cvtColor(image_data[:, :, :3], cv2.COLOR_BGR2GRAY)
                image_a = image_data[:, :, 3]
                image_data = image_data[:, :, :3]
            else:
                gray_image_data = np.squeeze(image_data, axis=-1)
                image_data = np.dstack((image_data, image_data, image_data))
            filled_edges = ColourLayer.detect_edges(gray_image_data, params.method, params.blur_radius, params.threshold)
            label_image_data = morphology.label(filled_edges, connectivity=2, background=0)

            features, bw_image_data = ColourLayer.get_object_features(gray_image_data, label_image_data, params)

            # sharpness analysis of the image using FFTs
            features = ColourLayer.sharpness_analysis(gray_image_data, image_data, features, params.estimate_sharpness)

            # mask the raw image with smoothed foreground mask
            blurd_bw_image_data = gaussian(bw_image_data, params.blur_radius)
            if np.max(blurd_bw_image_data) > 0:
                blurd_bw_image_data = blurd_bw_image_data / np.max(blurd_bw_image_data)
            for ind in range(3):
                image_data[:, :, ind] = image_data[:, :, ind] * blurd_bw_image_data

            # normalize the image as a float
            image_data = np.float32(image_data) if np.max(image_data) == 0 else np.float32(image_data) / np.max(image_data)

            image_data = ColourLayer.deconvolution(
                image_data,
                bw_image_data,
                blurd_bw_image_data,
                params.deconv,
                params.deconv_method,
                params.deconv_iter,
                params.deconv_mask_weight,
                params.small_float_val,
            )
            metadata["edge_detection"] = features
            if num_channels == NUM_CHANNELS_RGBA:
                image_data = np.dstack((image_data, image_a))
        except Exception as e:  # noqa: BLE001
            msg = f"Error applying edge detection: {e}"
            check_raise_error(params.raise_error, msg)
        return image_data, metadata

    @staticmethod
    def get_object_features(gray_image_data: np.ndarray, label_image_data: np.ndarray, params: EdgeDetectionParams) -> tuple[dict, np.ndarray]:
        """Get object features.

        Get the features of the object.

        Args:
            gray_image_data (np.ndarray): The grayscale image data.
            label_image_data (np.ndarray): The label image data.
            params (EdgeDetectionParams): The parameters for edge detection.

        Returns:
            tuple[dict, np.ndarray]: The features of the object and the binary image data.
        """
        props = measure.regionprops(label_image_data, gray_image_data)
        valid_object = False
        bw_image_data = None
        if len(props) > 0:
            max_area = 0
            max_area_ind = 0

            area_list = []

            for index, prop in enumerate(props):
                area_list.append(prop.axis_major_length)
                if prop.axis_major_length > max_area:
                    max_area = prop.axis_major_length
                    max_area_ind = index

            area_list = sorted(area_list, reverse=True)

            selected_index = max_area_ind

            if params.object_selection != "full_ROI" and params.object_type != "aggregate":
                bw_image_data = label_image_data == props[selected_index].label
            else:
                bw_image_data = label_image_data > 0
                # Recompute props on single mask
                props = measure.regionprops(bw_image_data.astype(np.uint8), gray_image_data)
                selected_index = 0

            bw = bw_image_data if np.max(bw_image_data) == 0 else bw_image_data / np.max(bw_image_data)

            features = {}
            clip_frac = float(np.sum(bw[:, 1]) + np.sum(bw[:, -2]) + np.sum(bw[1, :]) + np.sum(bw[-2, :])) / (2 * bw.shape[0] + 2 * bw.shape[1])

            # Save simple features of the object
            if params.object_selection != "full_ROI":
                selected_prop = props[selected_index]
                features.update(
                    {
                        "area": selected_prop.area,
                        "minor_axis_length": selected_prop.axis_minor_length,
                        "major_axis_length": selected_prop.axis_major_length,
                        "aspect_ratio": (
                            (selected_prop.axis_minor_length / selected_prop.axis_major_length) if selected_prop.axis_major_length != 0 else 1
                        ),
                        "orientation": selected_prop.orientation,
                    },
                )
            else:
                features.update(
                    {
                        "area": bw.shape[0] * bw.shape[1],
                        "minor_axis_length": min(bw.shape[0], bw.shape[1]),
                        "major_axis_length": max(bw.shape[0], bw.shape[1]),
                        "aspect_ratio": (
                            (props[selected_index].axis_minor_length / props[selected_index].axis_major_length)
                            if props[selected_index].axis_major_length != 0
                            else 1
                        ),
                        "orientation": 0,
                    },
                )

            # save all features except for those with  pixel data
            output_dict = {
                prop: props[selected_index][prop] for prop in props[selected_index] if prop not in ["convex_image", "filled_image", "image", "coords"]
            }
            features = output_dict
            features["clipped_fraction"] = clip_frac
            valid_object = True
        else:
            features = {
                "area": 0.0,
                "minor_axis_length": 0.0,
                "major_axis_length": 0.0,
                "aspect_ratio": 1,
                "orientation": 0.0,
                "clippped_fraction": 1.0,
            }
        features["valid_object"] = valid_object
        return (features, bw_image_data)

    @staticmethod
    def gaussian_psf(size: list[int], sigma: float) -> np.ndarray:
        """Gaussian point spread function.

        Create a Gaussian point spread function (PSF).

        Args:
            size (List[int]): The size of the PSF.
            sigma (float): The standard deviation of the PSF.

        Returns:
            np.ndarray: The Gaussian PSF.
        """
        if len(size) == NUM_DIMENSIONS_GREY:
            psf = np.zeros((size[0], size[1]))
            psf[size[0] // 2, size[1] // 2] = 1
        elif len(size) == NUM_CHANNELS_RGB:
            psf = np.zeros((size[0], size[1], size[2]))
            psf[size[0] // 2, size[1] // 2, size[2] // 2] = 1
        psf = gaussian(psf, sigma=sigma)
        psf /= psf.sum()
        return psf

    @staticmethod
    def motion_psf(size: list[float], length: float, angle_xy: float, angle_z: int = 0) -> np.ndarray:
        """Motion point spread function.

        Create a motion point spread function (PSF).

        Args:
            size (float[]): size of the PSF
            length (float): length of the PSF
            angle_xy (float): angle of the PSF
            angle_z (int, optional): tilt in the z-axis. Defaults to 0.

        Returns:
            np.ndarray: The motion PSF
        """
        if len(size) == NUM_DIMENSIONS_GREY:
            psf = np.zeros((size[0], size[1]))
            center_x = size[0] // 2
            center_y = size[1] // 2
            angle_rad = np.deg2rad(angle_xy)
            for i in range(length):
                x = int(center_x + i * np.cos(angle_rad))
                y = int(center_y + i * np.sin(angle_rad))
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    psf[x, y] = 1
        else:
            psf = np.zeros((size[0], size[1], size[2]))
            center_x = size[0] // 2
            center_y = size[1] // 2
            center_z = size[2] // 2
            angle_xy_rad = np.deg2rad(angle_xy)
            angle_z_rad = np.deg2rad(angle_z)

            for i in range(length):
                x = int(center_x + i * np.cos(angle_xy_rad) * np.cos(angle_z_rad))
                y = int(center_y + i * np.sin(angle_xy_rad) * np.cos(angle_z_rad))
                z = int(center_z + i * np.sin(angle_z_rad))
                if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
                    psf[x, y, z] = 1
        psf /= psf.sum()
        return psf

    @staticmethod
    def _white_balance(img: np.ndarray, num_channels: int) -> np.ndarray:
        """White balance.

        Perform white balancing on the image.

        Args:
            img (np.ndarray): The image to white balance.
            num_channels (int): The number of channels in the image.

        Returns:
            np.ndarray: The white balanced image.
        """
        r, g, b = cv2.split(img[:, :, :3])
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        avg_gray = (avg_r + avg_g + avg_b) / 3

        r_scale = avg_gray / avg_r
        g_scale = avg_gray / avg_g
        b_scale = avg_gray / avg_b

        r = cv2.convertScaleAbs(r * r_scale)
        g = cv2.convertScaleAbs(g * g_scale)
        b = cv2.convertScaleAbs(b * b_scale)

        return cv2.merge([r, g, b, img[..., 3]]) if num_channels == NUM_CHANNELS_RGBA else cv2.merge([r, g, b])

    @staticmethod
    def deconvolution(
        img: np.ndarray,
        bw_img: np.ndarray,
        blurd_bw_img: np.ndarray,
        deconv: bool,
        deconv_method: str,
        deconv_iter: int,
        deconv_mask_weight: float,
        small_float_val: float = 1e-6,
    ) -> np.ndarray:
        """Deconvolution.

        Perform deconvolution on the image.

        Args:
            img (np.ndarray): The image to deconvolve
            bw_img (np.ndarray): The binary image to use for deconvolution
            blurd_bw_img (np.ndarray): The blurred binary image to use for deconvolution
            deconv (bool): Whether to perform deconvolution
            deconv_method (str): The method to use for deconvolution
            deconv_iter (int): The number of iterations for deconvolution
            deconv_mask_weight (float): The weight for the deconvolution mask
            small_float_val (float, optional): The small float value. Defaults to 1e-6.

        Returns:
            np.ndarray: The deconvolved image
        """
        if deconv:
            # Get the intensity image in HSV space for sharpening
            with np.errstate(divide="ignore"):
                hsv_img = color.rgb2hsv(img)
            v_img = hsv_img[:, :, 2] * blurd_bw_img

            # Unsharp mask before masking with binary image
            if deconv_method == "UM":
                old_mean = np.mean(v_img)
                blurd = gaussian(v_img, 1.0)
                hpfilt = v_img - blurd * deconv_mask_weight
                v_img = hpfilt / (1 - deconv_mask_weight)

                new_mean = np.mean(v_img)
                if new_mean != 0:
                    v_img *= old_mean / new_mean

                v_img = np.clip(v_img, 0, 1)
                v_img = np.uint8(255 * v_img)

            # Resize bw_img to match v_img shape
            bw_img = resize(bw_img, v_img.shape)
            v_img[v_img == 0] = small_float_val

            # Richardson-Lucy deconvolution
            if deconv_method == "LR":
                psf = ColourLayer.make_gaussian(5, 3, center=None)
                v_img = restoration.richardson_lucy(v_img, psf, deconv_iter)

                v_img = np.clip(v_img, 0, None)
                v_img = np.uint8(255 * v_img / np.max(v_img) if np.max(v_img) != 0 else 255 * v_img)

            # Restore the RGB image from HSV
            v_img[v_img == 0] = small_float_val
            hsv_img[:, :, 2] = v_img
            img = color.hsv2rgb(hsv_img)

            # Restore img to 8-bit
            img_min = np.min(img)
            img_range = np.max(img) - img_min
            img = np.zeros(img.shape, dtype=np.uint8) if img_range == 0 else np.uint8(255 * (img - img_min) / img_range)
        else:
            # Restore img to 8-bit
            img = np.uint8(255 * img)

        return img

    @staticmethod
    def sharpness_analysis(
        gray_img: np.ndarray,
        img: np.ndarray,
        features: dict,
        estimate_sharpness: bool = True,
    ) -> dict:
        """Sharpness analysis.

        Estimate the sharpness of the image using FFTs.

        Args:
            gray_img (np.ndarray): The grayscale image
            img (np.ndarray): The image
            features (dict): The features of the image
            estimate_sharpness (bool, optional): Whether to estimate sharpness.
        Defaults to True.

        Returns:
            dict: The features of the image
        """
        if estimate_sharpness and features["valid_object"]:
            pad_size = 6
            max_dim = np.max(gray_img.shape)

            # Determine pad size for FFT
            for s in range(6, 15):
                if max_dim <= 2**s:
                    pad_r = 2**s - gray_img.shape[0]
                    pad_c = 2**s - gray_img.shape[1]
                    real_img = np.pad(gray_img, [(0, pad_r), (0, pad_c)], mode="constant")
                    pad_size = 2**s
                    break

            # Prefilter the image to remove some of the DC component
            real_img = real_img.astype("float") - np.mean(img)

            # Window the image to reduce ringing and energy leakage
            wind = ColourLayer.make_gaussian(pad_size, pad_size / 2, center=None)

            # Estimate blur of the image using the method from Roberts et al. 2011
            the_fft = np.fft.fft2(real_img * wind)
            fft_mag = np.abs(the_fft).astype("float")
            fft_mag = np.fft.fftshift(fft_mag)
            fft_mag = gaussian(fft_mag, 2)

            # Find all frequencies with energy above 5% of the max in the spectrum
            mask = fft_mag > 0.02 * np.max(fft_mag)

            rr, cc = np.nonzero(mask)
            rr = (rr.astype("float") - pad_size / 2) * 4 / pad_size
            cc = (cc.astype("float") - pad_size / 2) * 4 / pad_size
            features["sharpness"] = 1024 * np.max(np.sqrt(rr**2 + cc**2))
            return features
        features["sharpness"] = 0
        return features

    @staticmethod
    def detect_edges(img: np.ndarray, method: str, blur_radius: float, threshold: dict) -> np.ndarray:
        """Detect edges.

        Detect edges in the image.

        Args:
            img (np.ndarray): The image to detect edges
            method (str): The method to use for edge detection
            blur_radius (float): The radius for the blur
            threshold (dict): The threshold for edge detection

        Returns:
            np.ndarray: The filled edges
        """
        if method == "scharr":
            if len(img.shape) == NUM_CHANNELS_RGB:
                edges_mags = [scharr(img[:, :, i]) for i in range(NUM_CHANNELS_RGB)]
                filled_edges = [ColourLayer.process_edges(edges_mag, threshold["low"], blur_radius) for edges_mag in edges_mags]
            else:
                edges_mag = scharr(img)
                filled_edges = ColourLayer.process_edges(edges_mag, threshold["low"], blur_radius)
        elif method == "scharr_with_mean":
            if len(img.shape) == NUM_CHANNELS_RGB:
                edges_mags = [scharr(img[:, :, i]) for i in range(3)]
                filled_edges = [ColourLayer.process_edges_mean(edges_mag, blur_radius) for edges_mag in edges_mags]
            else:
                edges_mag = scharr(img)
                filled_edges = ColourLayer.process_edges_mean(edges_mag, blur_radius)
        elif method == "canny":
            if len(img.shape) == NUM_CHANNELS_RGB:
                edges = [cv2.Canny(img[:, :, i], threshold["low"], threshold["high"]) for i in range(NUM_CHANNELS_RGB)]
                filled_edges = [
                    morphology.erosion(
                        ndimage.binary_fill_holes(morphology.closing(edge, morphology.square(blur_radius))),
                        morphology.square(blur_radius),
                    )
                    for edge in edges
                ]
            else:
                edges = cv2.Canny(img, threshold["low"], threshold["high"])
                edges = morphology.closing(edges, morphology.square(blur_radius))
                filled_edges = ndimage.binary_fill_holes(edges)
                filled_edges = morphology.erosion(filled_edges, morphology.square(blur_radius))
        else:
            init_ls = checkerboard_level_set(img.shape[:2], 6)
            ls = morphological_chan_vese(
                img[:, :, 0] if len(img.shape) == NUM_CHANNELS_RGB else img,
                num_iter=11,
                init_level_set=init_ls,
                smoothing=3,
            )
            filled_edges = ls
        return filled_edges

    @staticmethod
    def process_edges(edges_mag: np.ndarray, low_threshold: float, blur_radius: float) -> np.ndarray:
        """Process the edges.

        Process the edges using the low threshold.

        Args:
            edges_mag (np.ndarray): The edges magnitude
            low_threshold (float): The low threshold
            blur_radius (float): The radius for the blur

        Returns:
            np.ndarray: The filled edges
        """
        edges_med = np.median(edges_mag)
        edges_thresh = low_threshold * edges_med
        edges = edges_mag >= edges_thresh
        edges = morphology.closing(edges, morphology.square(blur_radius))
        filled_edges = ndimage.binary_fill_holes(edges)
        return morphology.erosion(filled_edges, morphology.square(blur_radius))

    @staticmethod
    def process_edges_mean(edges_mag: np.ndarray, blur_radius: float) -> np.ndarray:
        """Process the edges.

        Process the edges using the mean.

        Args:
            edges_mag (np.ndarray): The edges magnitude
            blur_radius (float): The radius for the blur

        Returns:
            np.ndarray: The filled edges
        """
        edges_mean = np.mean(edges_mag)
        edges_std = np.std(edges_mag)
        edges_thresh = edges_mean + edges_std
        edges = edges_mag > edges_thresh
        edges = morphology.closing(edges, morphology.square(blur_radius))
        filled_edges = ndimage.binary_fill_holes(edges)
        return morphology.erosion(filled_edges, morphology.square(blur_radius))

    @staticmethod
    def make_gaussian(size: int, fwhm: int = 3, center: tuple | None = None) -> np.ndarray:  # noqa: ARG004
        """Make a square gaussian kernel.

        Method to make a square gaussian kernel.

        Args:
            size (int): The size of the square.
            fwhm (int, optional): The full-width-half-maximum. Defaults to 3.
            center (tuple, optional): The center of the square. Defaults to None.

        Returns:
            np.ndarray: The square gaussian kernel.
        """
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        # if center is None:
        x0 = y0 = size // 2
        # else:
        #     x0 = center[0]
        #     y0 = center[1]

        output = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)
        return output / np.sum(output)


ColorLayer = ColourLayer
