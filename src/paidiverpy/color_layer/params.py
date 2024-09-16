""" This module contains the dataclasses for the parameters of the color layer functions. """

from dataclasses import dataclass


@dataclass
class GrayScaleParams:
    """This class contains the parameters for the grayscale conversion."""

    pass


@dataclass
class GaussianBlurParams:
    """This class contains the parameters for the Gaussian blur."""

    sigma: float = 1.0


@dataclass
class EdgeDetectionParams:
    """This class contains the parameters for the edge detection."""

    method: str = "sobel"
    blur_radius: float = 1.0
    threshold: float = 0.1
    object_type: str = "bright"
    object_selection: str = "largest"
    estimate_sharpness: bool = False
    deconv: bool = False
    deconv_method: str = "richardson_lucy"
    deconv_iter: int = 10
    deconv_mask_weight: float = 0.03
    small_float_val: float = 1e-6


@dataclass
class SharpenParams:
    """This class contains the parameters for the sharpening."""

    alpha: float = 1.5
    beta: float = -0.5


@dataclass
class ContrastAdjustmentParams:
    """This class contains the parameters for the contrast adjustment"""

    method: str = "clahe"
    kernel_size: int = None
    clip_limit: float = 0.01
    gamma_value: float = 0.5


@dataclass
class IlluminationCorrectionParams:
    """This class contains the parameters for the illumination correction"""

    method: str = "rolling"
    radius: int = 100


@dataclass
class DeblurParams:
    """This class contains the parameters for the deblurring"""

    method: str = "wiener"
    psf_type: str = "gaussian"
    sigma: float = 20
    angle: int = 45


@dataclass
class ColourAlterationParams:
    """This class contains the parameters for the colour alteration"""

    method: str = "white-balance"


COLOR_LAYER_METHODS = {
    "grayscale": {"params": GrayScaleParams, "method": "grayscale"},
    "gaussian_blur": {"params": GaussianBlurParams, "method": "gaussian_blur"},
    "edge_detection": {"params": EdgeDetectionParams, "method": "edge_detection"},
    "sharpen": {"params": SharpenParams, "method": "sharpen"},
    "contrast": {"params": ContrastAdjustmentParams, "method": "contrast_adjustment"},
    "deblur": {"params": DeblurParams, "method": "deblur"},
    "illumination_correction": {
        "params": IlluminationCorrectionParams,
        "method": "illumination_correction",
    },
    "colour_alteration": {"params": ColourAlterationParams, "method": "colour_alteration"},
}
