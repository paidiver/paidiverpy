
"""Color layer module.

This module contains the ColorLayer class for processing the images in the
color layer.
"""

import logging
import cv2
import dask
import dask.array as da
import numpy as np
from dask import compute
from dask.diagnostics import ProgressBar
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
from paidiverpy.config.color_params import COLOR_LAYER_METHODS
from paidiverpy.config.color_params import ContrastAdjustmentParams
from paidiverpy.config.color_params import DeblurParams
from paidiverpy.config.color_params import EdgeDetectionParams
from paidiverpy.config.color_params import GaussianBlurParams
from paidiverpy.config.color_params import GrayScaleParams
from paidiverpy.config.color_params import IlluminationCorrectionParams
from paidiverpy.config.color_params import SharpenParams
from paidiverpy.config.config import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils import DynamicConfig
from paidiverpy.utils import raise_value_error

NUM_CHANNELS_RGB = 3
NUM_CHANNELS_RGBA = 4
NUM_IMAGE_DIMS = 2
DEFAULT_BITS = 8

class ColorLayer(Paidiverpy):
    """ColorLayer class.

    Process the images in the color layer.

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
        """Color layer run method.

        Run the color layer steps on the images based on the configuration
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
        method, params = self._get_method_by_mode(params, COLOR_LAYER_METHODS, mode)
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

        Method to process the images sequentially.

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

        Method to process the images in parallel.

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

    def _apply_grayscale_conversion(self,
                                    image_data: np.ndarray,
                                    params: GrayScaleParams) -> np.ndarray:
        """GrayScale conversion.

        Apply the grayscale conversion method specified by params

        Args:
            image_data (np.ndarray): The input image.
            params (GrayScaleParams): Parameters for the grayscale conversion.

        Returns:
            np.ndarray: The grayscale image.
        """
        if params.method == "average":
            return np.mean(image_data, axis=-1)
        if params.method == "luminosity":
            band1 = 0.2126 * image_data[..., 0]
            band2 = 0.7152 * image_data[..., 1]
            band3 = 0.0722 * image_data[..., 2]
            return band1 + band2 + band3
        if params.method == "desaturation":
            return (np.max(image_data, axis=-1) + np.min(image_data, axis=-1)) / 2
        return cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    def grayscale(self, image_data: np.ndarray, params: GrayScaleParams = None) -> np.ndarray:
        """Convert the image to grayscale.

        Method to convert the image to grayscale.

        Args:
            image_data (np.ndarray): The input image.
            params (GrayScaleParams, optional): Parameters for the grayscale conversion.

        Raises:
            ValueError: If the input image does not have 3 channels or 4 channels with alpha.

        Returns:
            np.ndarray: The grayscale image.
        """
        if params is None:
            params = GrayScaleParams()
        if (len(image_data.shape) == NUM_IMAGE_DIMS or
            (image_data.shape[-1] != NUM_CHANNELS_RGB and
            image_data.shape[-1] != NUM_CHANNELS_RGBA)):
            self.logger.error("Input image must have 3 or 4 channels in the last dimension.")
            if self.raise_error:
                msg = "Input image must have 3 or 4 channels in the last dimension."
                raise ValueError(msg)
            return image_data
        try:
            if params.keep_alpha and image_data.shape[-1] == NUM_CHANNELS_RGBA:
                alpha_channel = image_data[..., NUM_CHANNELS_RGBA-1]
                image_data = image_data[..., :NUM_CHANNELS_RGB]
            image_data = self._apply_grayscale_conversion(image_data, params)

            if params.invert_colors:
                image_data = 255 - image_data

            if params.keep_alpha and "alpha_channel" in locals():
                image_data = np.dstack([image_data, alpha_channel])

        except Exception as e:
            self.logger.exception("Error converting image to grayscale: %s", e)
            if self.raise_error:
                msg = f"Error converting image to grayscale: {e}"
                raise ValueError(msg) from e

        return image_data

    def gaussian_blur(self,
                      image_data: np.ndarray,
                      params: GaussianBlurParams = None) -> np.ndarray:
        """Gaussian blur.

        Method to apply Gaussian blur to the image.

        Args:
            image_data (np.ndarray): The image to apply Gaussian blur.
            params (GaussianBlurParams, optional): the parameters for the method.
        Defaults to GaussianBlurParams().

        Raises:
            ValueError: Error applying Gaussian blur.

        Returns:
            np.ndarray: The image in grayscale.
        """
        if params is None:
            params = GaussianBlurParams()
        try:
            image_data = cv2.GaussianBlur(image_data, (0, 0), params.sigma)
        except Exception as e:
            self.logger.exception("Error applying Gaussian blur: %s", e)
            if self.raise_error:
                msg = f"Error applying Gaussian blur: {e}"
                raise ValueError(msg) from e
        return image_data

    def sharpen(self, image_data: np.ndarray, params: SharpenParams = None) -> np.ndarray:
        """Sharpening.

        Method to apply sharpening to the image.

        Args:
            image_data (np.ndarray): The image to apply sharpening.
            params (SharpenParams, optional): Params for method. Defaults to SharpenParams().

        Raises:
            ValueError: Error applying sharpening.

        Returns:
            np.ndarray: The image with sharpening applied.
        """
        if params is None:
            params = SharpenParams()
        try:
            bits = image_data.dtype.itemsize * DEFAULT_BITS
            image_data = unsharp_mask(image_data, radius=params.alpha, amount=params.beta)
            multiply_factor = 255 if bits == DEFAULT_BITS else 65535
            image_data = np.clip(image_data * multiply_factor, 0, multiply_factor).astype(
                np.uint8 if bits == DEFAULT_BITS else np.uint16,
            )
        except Exception as e:
            self.logger.exception("Error applying sharpening: %s", e)
            if self.raise_error:
                msg = f"Error applying sharpening: {e}"
                raise ValueError(msg) from e
        return image_data

    def contrast_adjustment(
        self,
        image_data: np.ndarray,
        params: ContrastAdjustmentParams = None,
    ) -> np.ndarray:
        """Contrast adjustment.

        Method to apply contrast adjustment to the image.

        Args:
            image_data (np.ndarray): The image to apply contrast adjustment.
            params (ContrastAdjustmentParams, optional): Params for method.
        Defaults to ContrastAdjustmentParams().

        Raises:
            ValueError: Error applying contrast adjustment.

        Returns:
            np.ndarray: The image with contrast adjustment applied.
        """
        if params is None:
            params = ContrastAdjustmentParams()
        try:
            method = params.method
            kernel_size = tuple(params.kernel_size) if params.kernel_size else None
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
        except Exception as e:
            self.logger.exception("Error applying contrast adjustment: %s", e)
            if self.raise_error:
                msg = f"Error applying contrast adjustment: {e}"
                raise ValueError(msg) from e

        return image_data

    def illumination_correction(self,
                                image_data: np.ndarray,
                                params: IlluminationCorrectionParams = None,
    ) -> np.ndarray:
        """Illumination correction.

        Method to apply illumination correction to the image.

        Args:
            image_data (np.ndarray): The image to apply illumination correction.
            params (IlluminationCorrectionParams, optional): Params for method.
        Defaults to IlluminationCorrectionParams().

        Raises:
            ValueError: Error applying illumination correction.

        Returns:
            np.ndarray: The image with illumination correction applied.
        """
        if params is None:
            params = IlluminationCorrectionParams()
        try:
            method = params.method
            radius = params.radius

            if method == "rolling":
                background = rolling_ball(image_data, radius=radius)
                image_data = image_data - background

        except Exception as e:
            self.logger.exception("Error applying contrast adjustment: %s", e)
            if self.raise_error:
                msg = f"Error applying contrast adjustment: {e}"
                raise ValueError(msg) from e
        return image_data

    def deblur(self, image_data: np.ndarray, params: DeblurParams = None) -> np.ndarray:
        """Deblurring.

        Method to apply deblurring to the image.

        Args:
            image_data (np.ndarray): The image to apply deblurring.
            params (DeblurParams, optional): Params for method.
        Defaults to DeblurParams().

        Raises:
            ValueError: Unknown PSF type. Please use 'gaussian' or 'motion'.
            ValueError: Unknown method type. Please use 'wiener'.
            NotImplementedError: Unknown method type. Please use 'wiener'.
            ValueError: Error applying contrast adjustment.

        Returns:
            np.ndarray: The image with deblurring applied.
        """
        if params is None:
            params = DeblurParams()
        try:
            method = params.method
            psf_type = params.psf_type
            sigma = params.sigma
            angle = params.angle
            if method == "wiener":
                if psf_type == "gaussian":
                    psf = ColorLayer.gaussian_psf(size=image_data.shape[0], sigma=sigma)
                elif psf_type == "motion":
                    psf = ColorLayer.motion_psf(size=image_data.shape[0], length=sigma, angle=angle)
                else:
                    msg = "Unknown PSF type. Please use 'gaussian' or 'motion'."
                    raise_value_error(msg)
                bits = image_data.dtype.itemsize * DEFAULT_BITS
                if image_data.shape[-1] == 1:
                    image_data = np.squeeze(image_data)
                image_data = wiener(image_data, psf, balance=0.1)
                multiply_factor = 255 if bits == DEFAULT_BITS else 65535
                image_data = np.clip(image_data * multiply_factor, 0, multiply_factor).astype(
                    np.uint8 if bits == DEFAULT_BITS else np.uint16,
                )

            else:
                self.logger.error("Unknown method type. Please use 'wiener'.")
                if self.raise_error:
                    msg = "Unknown method type. Please use 'wiener'."
                    raise_value_error(msg)

        except Exception as e:
            self.logger.exception("Error applying contrast adjustment: %s", e)
            if self.raise_error:
                msg = f"Error applying contrast adjustment: {e}"
                raise_value_error(msg)
        return image_data

    def edge_detection(
        self,
        image_data: np.ndarray,
        params: EdgeDetectionParams = None,
    ) -> np.ndarray:
        """Edge detection.

        Method to apply edge detection to the image.

        Args:
            image_data (np.ndarray): The image to apply edge detection.
            params (EdgeDetectionParams, optional): Params for method.
        Defaults to EdgeDetectionParams().

        Raises:
            e: Error applying edge detection.

        Returns:
            np.ndarray: The image with edge detection applied.
        """
        if params is None:
            params = EdgeDetectionParams()
        try:
            if params.method == "sobel":
                return cv2.Sobel(image_data, cv2.CV_64F, 1, 1, ksize=5), None

            if len(image_data.shape) == NUM_CHANNELS_RGB and image_data.shape[-1] == NUM_CHANNELS_RGB:
                gray_image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray_image_data = image_data
                image_data = np.dstack((image_data, image_data, image_data))
            filled_edges = ColorLayer.detect_edges(gray_image_data, params.method, params.blur_radius, params.threshold)
            label_image_data = morphology.label(filled_edges, connectivity=2, background=0)
            props = measure.regionprops(label_image_data, gray_image_data)

            valid_object = False
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

                if params.object_selection != "Full ROI" and params.object_type != "Aggregate":
                    bw_image_data = label_image_data == props[selected_index].label
                else:
                    bw_image_data = label_image_data > 0
                    # Recompute props on single mask
                    props = measure.regionprops(bw_image_data.astype(np.uint8), gray_image_data)
                    selected_index = 0

                bw = bw_image_data if np.max(bw_image_data) == 0 else bw_image_data / np.max(bw_image_data)

                features = {}
                clip_frac = float(np.sum(bw[:, 1]) + np.sum(bw[:, -2]) + np.sum(bw[1, :]) + np.sum(bw[-2, :])) / (
                    2 * bw.shape[0] + 2 * bw.shape[1]
                )

                # Save simple features of the object
                if params.object_selection != "Full ROI":
                    selected_prop = props[selected_index]
                    features.update(
                        {
                            "area": selected_prop.area,
                            "minor_axis_length": selected_prop.axis_minor_length,
                            "major_axis_length": selected_prop.axis_major_length,
                            "aspect_ratio": (
                                (selected_prop.axis_minor_length / selected_prop.axis_major_length)
                                if selected_prop.axis_major_length != 0
                                else 1
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
                    prop: props[selected_index][prop]
                    for prop in props[selected_index]
                    if prop not in ["convex_image", "filled_image", "image", "coords"]
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

            # sharpness analysis of the image using FFTs
            features = ColorLayer.sharpness_analysis(gray_image_data, image_data, features, params.estimate_sharpness)

            # mask the raw image with smoothed foreground mask
            blurd_bw_image_data = gaussian(bw_image_data, params.blur_radius)
            if np.max(blurd_bw_image_data) > 0:
                blurd_bw_image_data = blurd_bw_image_data / np.max(blurd_bw_image_data)
            for ind in range(3):
                image_data[:, :, ind] = image_data[:, :, ind] * blurd_bw_image_data

            # normalize the image as a float
            if np.max(image_data) == 0:
                image_data = np.float32(image_data)
            else:
                image_data = np.float32(image_data) / np.max(image_data)

            image_data = ColorLayer.deconvolution(
                image_data,
                bw_image_data,
                blurd_bw_image_data,
                params.deconv,
                params.deconv_method,
                params.deconv_iter,
                params.deconv_mask_weight,
                params.small_float_val,
            )
        except Exception as e:
            self.logger.exception("Error applying edge detection: %s", e)
            if self.raise_error:
                msg = f"Error applying edge detection: {e}"
                raise ValueError(msg) from e
        return image_data, features

    @staticmethod
    def gaussian_psf(size: int, sigma: float) -> np.ndarray:
        """Gaussian point spread function.

        Create a Gaussian point spread function (PSF).

        Args:
            size (int): The size of the PSF.
            sigma (float): The standard deviation of the PSF.

        Returns:
            np.ndarray: The Gaussian PSF.
        """
        psf = np.zeros((size, size))
        psf[size // 2, size // 2] = 1
        psf = gaussian(psf, sigma=sigma)
        psf /= psf.sum()
        return psf

    @staticmethod
    def motion_psf(size: float, length: float, angle: float) -> np.ndarray:
        """Motion point spread function.

        Create a motion point spread function (PSF).

        Args:
            size (float): size of the PSF
            length (float): length of the PSF
            angle (float): angle of the PSF

        Returns:
            np.ndarray: The motion PSF
        """
        psf = np.zeros((size, size))
        center = size // 2
        angle_rad = np.deg2rad(angle)
        for i in range(length):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1
        psf /= psf.sum()
        return psf

    @staticmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        """Normalize the image.

        Normalize the image to [0, 1].

        Args:
            img (np.ndarray): The image to normalize

        Returns:
            np.ndarray: The normalized image
        """
        min_val = np.min(img)
        max_val = np.max(img)

        # Shift and scale the image to [0, 1]
        return (img - min_val) / (max_val - min_val)


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
            if deconv_method.lower() == "um":
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
            if deconv_method.lower() == "lr":
                psf = ColorLayer.make_gaussian(5, 3, center=None)
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
            wind = ColorLayer.make_gaussian(pad_size, pad_size / 2, center=None)

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
    def detect_edges(img: np.ndarray,
                     method: str,
                     blur_radius: float,
                     threshold: tuple) -> np.ndarray:
        """Detect edges.

        Detect edges in the image.

        Args:
            img (np.ndarray): The image to detect edges
            method (str): The method to use for edge detection
            blur_radius (float): The radius for the blur
            threshold (tuple): The threshold for edge detection

        Returns:
            np.ndarray: The filled edges
        """
        if method == "Scharr":
            if len(img.shape) == NUM_CHANNELS_RGB:
                edges_mags = [scharr(img[:, :, i]) for i in range(NUM_CHANNELS_RGB)]
                filled_edges = [
                    ColorLayer.process_edges(edges_mag, threshold[0], blur_radius) for edges_mag in edges_mags
                ]
            else:
                edges_mag = scharr(img)
                filled_edges = ColorLayer.process_edges(edges_mag, threshold[0], blur_radius)
        elif method == "Scharr-with-mean":
            if len(img.shape) == NUM_CHANNELS_RGB:
                edges_mags = [scharr(img[:, :, i]) for i in range(3)]
                filled_edges = [ColorLayer.process_edges_mean(edges_mag, blur_radius) for edges_mag in edges_mags]
            else:
                edges_mag = scharr(img)
                filled_edges = ColorLayer.process_edges_mean(edges_mag, blur_radius)
        elif method == "Canny":
            if len(img.shape) == NUM_CHANNELS_RGB:
                edges = [cv2.Canny(img[:, :, i], threshold[0], threshold[1]) for i in range(NUM_CHANNELS_RGB)]
                filled_edges = [
                    morphology.erosion(
                        ndimage.binary_fill_holes(morphology.closing(edge, morphology.square(blur_radius))),
                        morphology.square(blur_radius),
                    )
                    for edge in edges
                ]
            else:
                edges = cv2.Canny(img, threshold[0], threshold[1])
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
    def make_gaussian(size: int, fwhm: int = 3, center: tuple | None = None) -> np.ndarray:
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

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        output = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)
        return output / np.sum(output)
