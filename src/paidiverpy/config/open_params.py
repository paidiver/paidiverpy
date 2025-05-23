"""Custom parameters dataclasses.

This module contains the dataclasses for the parameters used in the custom_params module.
"""

from typing import Literal
from pydantic import Field
from pydantic import model_validator
from paidiverpy.utils.base_model import BaseModel

SUPPORTED_OPENCV_IMAGE_TYPES = {
    "bmp",
    "dib",  # Windows bitmaps
    "jpg",
    "jpeg",
    "jpe",  # JPEG files
    "jp2",  # JPEG 2000 files
    "png",  # Portable Network Graphics
    "webP",  # WebP
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",  # Portable image format
    "sr",
    "ras",  # Sun rasters
    "tiff",
    "tif",  # TIFF files
    "exr",  # OpenEXR Image files
    "hdr",
    "pic",  # Radiance HDR
    "",  # non specified
}

SUPPORTED_PIL_IMAGE_TYPES = {
    "bmp",
    "dib",  # Windows bitmaps
    "jpg",
    "jpeg",  # JPEG files
    "jp2",  # JPEG 2000 files
    "png",  # Portable Network Graphics
    "ppm",
    "pgm",
    "pbm",  # Portable image format
    "tiff",
    "tif",  # TIFF files
    "webp",  # WebP
    "",  # non specified
}

SUPPORTED_EXIF_IMAGE_TYPES = SUPPORTED_PIL_IMAGE_TYPES | {
    "nef"  # Nikon RAW
}


SUPPORTED_RAWPY_IMAGE_TYPES = {
    "crw",  # Canon RAW
    "cr2",  # Canon RAW
    "cr3",  # Canon RAW
    "dng",  # Adobe Digital Negative
    "nef",  # Nikon RAW
    "nrw",  # Nikon RAW
    "orf",  # Olympus RAW
    "rw2",  # Panasonic RAW
    "raf",  # Fuji RAW
}


class ImageOpenArgsRawPyParams(BaseModel):
    """Parameters for RawPy postprocessing (rawpy.RawPy.postprocess)."""

    demosaic_algorithm: int | None = Field(None, description="Demosaicing algorithm (e.g. rawpy.DemosaicAlgorithm.AHD)")
    half_size: bool = Field(default=False, description="Reduce each 2x2 block to one pixel (half-size output)")
    four_color_rgb: bool = Field(default=False, description="Use separate interpolations for two green channels")
    dcb_iterations: int = Field(default=0, description="Number of DCB correction passes")
    dcb_enhance: bool = Field(default=False, description="Enhanced DCB interpolation colors")
    fbdd_noise_reduction: int = Field(default=0, description="FBDD noise reduction mode (0=Off)")
    noise_thr: float | None = Field(default=None, description="Threshold for wavelet denoising")
    median_filter_passes: int = Field(default=0, description="Median filter passes after demosaicing")
    use_camera_wb: bool = Field(default=False, description="Use camera white balance")
    use_auto_wb: bool = Field(default=False, description="Use automatic white balance")
    user_wb: list[float] | None = Field(default=None, description="Manual white balance multipliers [R, G1, G2, B]")
    output_color: int = Field(default=1, description="Output color space (e.g. rawpy.ColorSpace.sRGB = 1)")
    output_bps: int = Field(default=8, description="Bits per sample in output image (8 or 16)")
    user_flip: int | None = Field(default=None, description="Image flip/orientation override")
    user_black: int | None = Field(default=None, description="Override black level")
    user_sat: int | None = Field(default=None, description="Override saturation (white level)")
    no_auto_bright: bool = Field(default=False, description="Disable automatic brightness scaling")
    auto_bright_thr: float | None = Field(default=None, description="Threshold for clipping in auto brightness")
    adjust_maximum_thr: float = Field(default=0.75, description="Maximum threshold adjustment factor")
    bright: float = Field(default=1.0, description="Brightness scaling factor")
    highlight_mode: int = Field(default=0, description="Highlight handling mode (e.g. rawpy.HighlightMode.Clip = 0)")
    exp_shift: float | None = Field(default=None, description="Linear exposure shift (0.25 to 8.0)")
    exp_preserve_highlights: float = Field(default=0.0, description="Highlight preservation during exposure adjustment")
    no_auto_scale: bool = Field(default=False, description="Disable automatic pixel value scaling")
    gamma: tuple[float, float] | None = Field(default=None, description="Gamma correction parameters (power, slope)")
    chromatic_aberration: tuple[float, float] | None = Field(default=None, description="Red and blue scale correction")
    bad_pixels_path: str | None = Field(default=None, description="Path to bad pixel file for correction")


class ImageOpenArgsRawParams(BaseModel):
    """Parameters for manually loading raw images with specific metadata.

    These parameters are required when the image format is not supported by standard libraries.
    """

    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    bit_depth: Literal[8, 16] = Field(..., description="Bit depth of the image: 8 or 16.")

    endianness: Literal["little", "big"] | None = Field(default=None, description="Endianness of the image data. Only applicable to 16-bit images.")

    layout: Literal["5:6:5", "5:5:5", "6:5:5", "5:5:6"] | None = Field(
        default="5:6:5", description="RGB layout format (e.g. RGB565). Only for 16-bit images."
    )

    image_misc: str = Field(default="", description="Comma-separated string for image flags (e.g., 'bayer,vertical_flip').")

    bayer_pattern: Literal["GB", "RG", "BG", "GR"] | None = Field(
        default=None, description="Bayer pattern (e.g., GB, RG, BG, GR). Only for 8-bit bayer images."
    )

    file_header_size: int = Field(default=0, description="Number of bytes to skip at the beginning of the file.")

    swap_bytes: bool | None = Field(default=False, description="Swap bytes for endianness conversion. Only applicable for 16-bit images.")

    channels: int = Field(default=1, description="Number of channels in the image. Default is 1 (grayscale).")


class ImageOpenArgsOpenCVParams(BaseModel):
    """Parameters for OpenCV image loading."""

    dtype: str = Field(default="uint8", description="Data type of the image (e.g., 'uint8', 'float32')")
    flags: int = Field(default=-1, description="OpenCV flags for image loading (e.g., cv2.IMREAD_COLOR). Default is -1 for loading the image as is.")


class ImageOpenArgs(BaseModel):
    """Wrapper for specifying image format and associated parameters."""

    params: ImageOpenArgsRawPyParams | ImageOpenArgsRawParams | ImageOpenArgsOpenCVParams | dict = Field(
        default_factory=dict, description="Parameters for the image"
    )

    @model_validator(mode="after")
    def validate_params(self) -> "ImageOpenArgs":
        """Validate `params` based on `image_type` and cast to appropriate type."""
        if isinstance(self.params, dict):
            img_type = self.image_type.lower()
            if img_type in SUPPORTED_RAWPY_IMAGE_TYPES:
                self.params = ImageOpenArgsRawPyParams(**self.params)
            elif img_type in SUPPORTED_OPENCV_IMAGE_TYPES:
                self.params = ImageOpenArgsOpenCVParams(**self.params)
            else:
                self.params = ImageOpenArgsRawParams(**self.params)
        return self
