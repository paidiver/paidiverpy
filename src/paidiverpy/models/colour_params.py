"""Colour layer parameters models."""

from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class GrayScaleParams(BaseModel):
    """Parameters for the grayscale conversion."""

    method: str = Field("opencv", description="Grayscale conversion method")
    invert_colours: bool = Field(False, description="Invert grayscale values")
    raise_error: bool = Field(False, description="Raise error on failure")


class GaussianBlurParams(BaseModel):
    """Parameters for Gaussian blur."""

    sigma: float = Field(1.0, description="Sigma value for Gaussian kernel")
    raise_error: bool = Field(False, description="Raise error on failure")


class SharpenParams(BaseModel):
    """Parameters for sharpening."""

    alpha: float = Field(1.5, description="Weight of original image")
    beta: float = Field(-0.5, description="Weight of blurred image")
    raise_error: bool = Field(False, description="Raise error on failure")


class ContrastAdjustmentParams(BaseModel):
    """Parameters for contrast adjustment."""

    method: str = Field("clahe", description="Contrast adjustment method")
    kernel_size: dict | int | None = Field(
        None,
        description=(
            "Kernel size for CLAHE. It can be a dict with the format "
            "{'dim1': value, 'dim2': value} until the ndim, or an integer for square kernel size."
        ),
    )
    clip_limit: float = Field(0.01, description="Clip limit for CLAHE")
    gamma_value: float = Field(0.5, description="Gamma correction value")
    raise_error: bool = Field(False, description="Raise error on failure")


class IlluminationCorrectionParams(BaseModel):
    """Parameters for illumination correction."""

    method: str = Field("rolling", description="Correction method (e.g. rolling ball)")
    radius: int = Field(5, description="Radius of the illumination filter")
    raise_error: bool = Field(False, description="Raise error on failure")


class DeblurParams(BaseModel):
    """Parameters for deblurring."""

    method: str = Field("wiener", description="Deblurring method")
    psf_type: str = Field("gaussian", description="Point Spread Function type")
    sigma: int = Field(20, description="Sigma for Gaussian PSF")
    angle: int = Field(45, description="Angle for motion blur PSF")
    raise_error: bool = Field(False, description="Raise error on failure")


class ColourAlterationParams(BaseModel):
    """Parameters for colour alteration."""

    method: str = Field("white_balance", description="Colour alteration method")
    raise_error: bool = Field(False, description="Raise error on failure")


class EdgeDetectionParams(BaseModel):
    """Parameters for edge detection."""

    method: str = Field("sobel", description="Edge detection method")
    blur_radius: int = Field(1, description="Blur radius before edge detection")
    threshold: dict | None = Field(
        None, description="Threshold for edge detection. It should have the format {'low': value, 'high': value}. High value is optional."
    )
    object_type: str = Field("bright", description="Type of object (bright or dark)")
    object_selection: str = Field("largest", description="Object selection strategy")
    estimate_sharpness: bool = Field(False, description="Estimate image sharpness")
    deconv: bool = Field(False, description="Apply deconvolution")
    deconv_method: str = Field("LR", description="Deconvolution method")
    deconv_iter: int = Field(10, description="Deconvolution iterations")
    deconv_mask_weight: float = Field(0.03, description="Deconvolution mask weighting")
    small_float_val: float = Field(1e-6, description="Small float to avoid division by zero")
    raise_error: bool = Field(False, description="Raise error on failure")


COLOUR_LAYER_METHODS = {
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

ColourParamsUnion = (
    GrayScaleParams
    | GaussianBlurParams
    | EdgeDetectionParams
    | SharpenParams
    | ContrastAdjustmentParams
    | DeblurParams
    | IlluminationCorrectionParams
    | ColourAlterationParams
    | EdgeDetectionParams
    | dict
)
