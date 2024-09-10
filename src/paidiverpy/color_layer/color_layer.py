""" Open raw image file
"""

import cv2
import numpy as np
from skimage import morphology, measure, restoration
from skimage import color
from skimage.filters import scharr, gaussian, unsharp_mask
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import resize
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from scipy import ndimage
from skimage.restoration import rolling_ball, wiener
from skimage.util import img_as_float
from tqdm import tqdm
from paidiverpy import Paidiverpy
from paidiverpy.image_layer import ImageLayer
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.color_layer.params import (
    COLOR_LAYER_METHODS,
    DeblurParams,
    GrayScaleParams,
    GaussianBlurParams,
    EdgeDetectionParams,
    IlluminationCorrectionParams,
    SharpenParams,
    ContrastAdjustmentParams,
)


class ColorLayer(Paidiverpy):
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

    def run(self):
        mode = self.step_metadata.get("mode")
        if not mode:
            raise ValueError("The mode is not defined in the configuration file.")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}

        if mode not in COLOR_LAYER_METHODS:
            raise ValueError(f"Unsupported mode: {mode}")

        params = COLOR_LAYER_METHODS[mode]["params"](**params)

        method_info = COLOR_LAYER_METHODS[mode]
        method_name = method_info["method"]

        images = self.images.get_step(step=len(self.images.images) - 1, by_order=True)
        image_list = []
        features = None

        for index, img in tqdm(
            enumerate(images), total=len(images), desc="Processing Images"
        ):
            img_data = img.image
            method = getattr(self, method_name)
            if method_name == "edge_detection":
                img_data, features = method(img_data, params=params)
            else:
                img_data = method(img_data, params=params)

            catalog = self.get_catalog(flag="all").iloc[index].to_dict()
            if features:
                catalog.update(features)
            img = ImageLayer(
                image=img_data,
                image_metadata=catalog,
                step_order=self.images.get_last_step_order(),
                step_name=self.step_name,
            )
            image_list.append(img)

        if not test:
            self.step_name = (
                f"convert_{self.config_index}" if not self.step_name else self.step_name
            )
            self.images.add_step(
                step=self.step_name, images=image_list, step_metadata=self.step_metadata
            )

    def grayscale(self, img, params: GrayScaleParams = GrayScaleParams()):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.logger.error(f"Error converting image to grayscale: {e}")
            if self.raise_error:
                raise e
        return img

    def gaussian_blur(
        self, img, params: GaussianBlurParams = GaussianBlurParams()
    ):
        try:
            img = cv2.GaussianBlur(img, (0, 0), params.sigma)
        except Exception as e:
            self.logger.error(f"Error applying Gaussian blur: {e}")
            if self.raise_error:
                raise e
        return img

    def sharpen(self, img, params: SharpenParams = SharpenParams()):
        try:
            img = unsharp_mask(img, radius=params.alpha, amount=params.beta)
        except Exception as e:
            self.logger.error(f"Error applying sharpening: {e}")
            if self.raise_error:
                raise e
        return img

    def contrast_adjustment(
        self,
        img,    
        params: ContrastAdjustmentParams = ContrastAdjustmentParams(),
    ):
        try:
            method = params.method
            kernel_size = tuple(params.kernel_size) if params.kernel_size else None
            clip_limit = params.clip_limit
            gamma_value = params.gamma_value

            if method == 'clahe':
                img = equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
            elif method == 'gamma':
                img = adjust_gamma(img, gamma=gamma_value)            

            # # normalize into original format
            # if img.dtype == 'uint8':
            #     output_bits = 8
            # elif img.dtype == 'uint16':
            #     output_bits = 16
            # else:
            #     output_bits = 8  # default
            # img_norm = ConvertLayer.convert_bits(img_adj, output_bits, logger=logger, autoscale=True)
            
        except Exception as e:
            self.logger.error(f"Error applying contrast adjustment: {e}")
            if self.raise_error:
                raise e
        return img

    def illumination_correction(self, img, params: IlluminationCorrectionParams = IlluminationCorrectionParams()):
        try:
            method = params.method
            radius = params.radius

            if method == 'rolling':
                background = rolling_ball(img, radius=radius)
                img = img - background

        except Exception as e:
            self.logger.error(f"Error applying contrast adjustment: {e}")
            if self.raise_error:
                raise e
        return img

    def deblur(self, img, params: DeblurParams = DeblurParams()):
        try:
            method = params.method
            psf_type = params.psf_type
            sigma = params.sigma
            size = params.size

            if method == 'wiener':
                if psf_type == 'gaussian':
                    psf = ColorLayer.gaussian_psf(size=size, sigma=sigma)
                elif psf_type == 'motion':
                    psf = ColorLayer.motion_psf(length=sigma)
                else:
                    raise ValueError("Unknown PSF type. Please use 'gaussian' or 'motion'.")

                img_float = img_as_float(img)
                img = wiener(img_float, psf, balance=0.1)

                # shift negative values [0, 1]
                # img = ColorLayer.normalize_img(img)
            else:
                raise ValueError("Unknown method type. Please use 'wiener'.")

            # # normalize into original format
            # if img.dtype == 'uint8':
            #     output_bits = 8
            # elif img.dtype == 'uint16':
            #     output_bits = 16
            # else:
            #     output_bits = 8  # default
            # img_norm = ConvertLayer.convert_bits(de_blurred_float, output_bits, logger=logger, autoscale=True)

        except Exception as e:
            self.logger.error(f"Error applying contrast adjustment: {e}")
            if self.raise_error:
                raise e
        return img

    def edge_detection(
        self, img, params: EdgeDetectionParams = EdgeDetectionParams()
    ):
        try:
            if params.method == "sobel":
                return cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5), None

            if len(img.shape) == 3:
                gray_img = ConvertLayer.channel_convert(
                    to="gray",
                    img=img,
                    logger=self.logger,
                    raise_error=self.raise_error,
                )
            else:
                gray_img = img
                img = np.dstack((img, img, img))
            filled_edges = ColorLayer.detect_edges(
                gray_img, params.method, params.blur_radius, params.threshold
            )
            label_img = morphology.label(filled_edges, connectivity=2, background=0)
            props = measure.regionprops(label_img, gray_img)

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

                if (
                    params.object_selection != "Full ROI"
                    and params.object_type != "Aggregate"
                ):
                    bw_img = label_img == props[selected_index].label
                else:
                    bw_img = label_img > 0
                    # Recompute props on single mask
                    props = measure.regionprops(bw_img.astype(np.uint8), gray_img)
                    selected_index = 0

                bw = bw_img if np.max(bw_img) == 0 else bw_img / np.max(bw_img)

                features = {}
                clip_frac = float(
                    np.sum(bw[:, 1])
                    + np.sum(bw[:, -2])
                    + np.sum(bw[1, :])
                    + np.sum(bw[-2, :])
                ) / (2 * bw.shape[0] + 2 * bw.shape[1])

                # Save simple features of the object
                if params.object_selection != "Full ROI":
                    selected_prop = props[selected_index]
                    features.update(
                        {
                            "area": selected_prop.area,
                            "minor_axis_length": selected_prop.axis_minor_length,
                            "major_axis_length": selected_prop.axis_major_length,
                            "aspect_ratio": (
                                (
                                    selected_prop.axis_minor_length
                                    / selected_prop.axis_major_length
                                )
                                if selected_prop.axis_major_length != 0
                                else 1
                            ),
                            "orientation": selected_prop.orientation,
                        }
                    )
                else:
                    features.update(
                        {
                            "area": bw.shape[0] * bw.shape[1],
                            "minor_axis_length": min(bw.shape[0], bw.shape[1]),
                            "major_axis_length": max(bw.shape[0], bw.shape[1]),
                            "aspect_ratio": (
                                (
                                    props[selected_index].axis_minor_length
                                    / props[selected_index].axis_major_length
                                )
                                if props[selected_index].axis_major_length != 0
                                else 1
                            ),
                            "orientation": 0,
                        }
                    )

                # save all features except for those with  pixel data
                output_dict = {
                    prop: props[selected_index][prop]
                    for prop in props[selected_index]
                    if prop
                    not in ["convex_image", "filled_image", "image", "coords"]
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
            features = ColorLayer.sharpness_analysis(
                gray_img, img, features, params.estimate_sharpness
            )

            # mask the raw image with smoothed foreground mask
            blurd_bw_img = gaussian(bw_img, params.blur_radius)
            if np.max(blurd_bw_img) > 0:
                blurd_bw_img = blurd_bw_img / np.max(blurd_bw_img)
            for ind in range(0, 3):
                img[:, :, ind] = img[:, :, ind] * blurd_bw_img

            # normalize the image as a float
            if np.max(img) == 0:
                img = np.float32(img)
            else:
                img = np.float32(img) / np.max(img)

            img = ColorLayer.deconvolution(
                img,
                bw_img,
                blurd_bw_img,
                params.deconv,
                params.deconv_method,
                params.deconv_iter,
                params.deconv_mask_weight,
                params.small_float_val,
            )
        except Exception as e:
            self.logger.error(f"Error applying edge detection: {e}")
            if self.raise_error:
                raise e
        return img, features

    @staticmethod
    def gaussian_psf(size, sigma):
        psf = np.zeros((size, size))
        psf[size // 2, size // 2] = 1
        psf = gaussian(psf, sigma=sigma)
        return psf / psf.sum()

    @staticmethod
    def motion_psf(length, angle=0):
        psf = np.zeros((length, length))
        psf[length // 2, :] = np.ones(length)
        psf = psf / psf.sum()
        return psf

    @staticmethod
    def normalize_img(img):
        min_val = np.min(img)
        max_val = np.max(img)

        # Shift and scale the image to [0, 1]
        norm_img = (img - min_val) / (max_val - min_val)

        return norm_img

    @staticmethod
    def deconvolution(img,
                      bw_img,
                      blurd_bw_img,
                      deconv,
                      deconv_method,
                      deconv_iter,
                      deconv_mask_weight,
                      small_float_val=1e-6):
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
                v_img = np.uint8(
                    255 * v_img / np.max(v_img) if np.max(v_img) != 0 else 255 * v_img
                )

            # Restore the RGB image from HSV
            v_img[v_img == 0] = small_float_val
            hsv_img[:, :, 2] = v_img
            img = color.hsv2rgb(hsv_img)

            # Restore img to 8-bit
            img_min = np.min(img)
            img_range = np.max(img) - img_min
            img = (
                np.zeros(img.shape, dtype=np.uint8)
                if img_range == 0
                else np.uint8(255 * (img - img_min) / img_range)
            )
        else:
            # Restore img to 8-bit
            img = np.uint8(255 * img)

        return img

    @staticmethod
    def sharpness_analysis(gray_img, img, features, estimate_sharpness=True):
        if estimate_sharpness:
            if features["valid_object"]:
                pad_size = 6
                max_dim = np.max(gray_img.shape)

                # Determine pad size for FFT
                for s in range(6, 15):
                    if max_dim <= 2**s:
                        pad_r = 2**s - gray_img.shape[0]
                        pad_c = 2**s - gray_img.shape[1]
                        real_img = np.pad(
                            gray_img, [(0, pad_r), (0, pad_c)], mode="constant"
                        )
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
    def detect_edges(img, method, blur_radius, threshold):
        if method == "Scharr":
            if len(img.shape) == 3:
                edges_mags = [scharr(img[:, :, i]) for i in range(3)]
                filled_edges = [
                    ColorLayer.process_edges(edges_mag, threshold[0], blur_radius)
                    for edges_mag in edges_mags
                ]
            else:
                edges_mag = scharr(img)
                filled_edges = ColorLayer.process_edges(
                    edges_mag, threshold[0], blur_radius
                )
        elif method == "Scharr-with-mean":
            if len(img.shape) == 3:
                edges_mags = [scharr(img[:, :, i]) for i in range(3)]
                filled_edges = [
                    ColorLayer.process_edges_mean(edges_mag, blur_radius)
                    for edges_mag in edges_mags
                ]
            else:
                edges_mag = scharr(img)
                filled_edges = ColorLayer.process_edges_mean(edges_mag, blur_radius)
        elif method == "Canny":
            if len(img.shape) == 3:
                edges = [
                    cv2.Canny(img[:, :, i], threshold[0], threshold[1])
                    for i in range(3)
                ]
                filled_edges = [
                    morphology.erosion(
                        ndimage.binary_fill_holes(
                            morphology.closing(edge, morphology.square(blur_radius))
                        ),
                        morphology.square(blur_radius),
                    )
                    for edge in edges
                ]
            else:
                edges = cv2.Canny(img, threshold[0], threshold[1])
                edges = morphology.closing(edges, morphology.square(blur_radius))
                filled_edges = ndimage.binary_fill_holes(edges)
                filled_edges = morphology.erosion(
                    filled_edges, morphology.square(blur_radius)
                )
        else:
            init_ls = checkerboard_level_set(img.shape[:2], 6)
            ls = morphological_chan_vese(
                img[:, :, 0] if len(img.shape) == 3 else img,
                num_iter=11,
                init_level_set=init_ls,
                smoothing=3,
            )
            filled_edges = ls
        return filled_edges

    @staticmethod
    def process_edges(edges_mag, low_threshold, blur_radius):
        edges_med = np.median(edges_mag)
        edges_thresh = low_threshold * edges_med
        edges = edges_mag >= edges_thresh
        edges = morphology.closing(edges, morphology.square(blur_radius))
        filled_edges = ndimage.binary_fill_holes(edges)
        filled_edges = morphology.erosion(filled_edges, morphology.square(blur_radius))
        return filled_edges

    @staticmethod
    def process_edges_mean(edges_mag, blur_radius):
        edges_mean = np.mean(edges_mag)
        edges_std = np.std(edges_mag)
        edges_thresh = edges_mean + edges_std
        edges = edges_mag > edges_thresh
        edges = morphology.closing(edges, morphology.square(blur_radius))
        filled_edges = ndimage.binary_fill_holes(edges)
        filled_edges = morphology.erosion(filled_edges, morphology.square(blur_radius))
        return filled_edges

    @staticmethod
    def make_gaussian(size, fwhm=3, center=None):
        """Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """

        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        output = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)
        output = output / np.sum(output)

        return output
