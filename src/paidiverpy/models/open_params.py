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

    class Config:
        """Configuration for the RawPy parameters model."""

        extra = "allow"


class ImageOpenArgsRawParams(BaseModel):
    """Parameters for manually loading raw images with specific metadata.

    These parameters are required when the image format is not supported by standard libraries.
    """

    width: int = Field(default=0, description="Image width in pixels.")
    height: int = Field(default=0, description="Image height in pixels.")
    bit_depth: Literal[8, 16] = Field(default=8, description="Bit depth of the image: 8 or 16.")

    endianness: Literal["little", "big"] | None = Field(default=None, description="Endianness of the image data. Only applicable to 16-bit images.")

    layout: Literal["5:6:5", "5:5:5", "6:5:5", "5:5:6"] | None = Field(
        default="5:6:5", description="RGB layout format (e.g. RGB565). Only for 16-bit images."
    )

    image_misc: str = Field(default="", description="Comma-separated string for image flags (e.g., 'bayer,vertical_flip').")

    bayer_pattern: Literal["GB", "RG", "BG", "GR"] | None = Field(
        default=None, description="Bayer pattern (e.g., GB, RG, BG, GR). Only for 8-bit bayer images."
    )

    file_header_size: int = Field(default=0, description="Number of bytes to skip at the beginning of the file.")

    swap_bytes: bool = Field(default=False, description="Swap bytes for endianness conversion. Only applicable for 16-bit images.")

    channels: int = Field(default=1, description="Number of channels in the image. Default is 1 (grayscale).")


class ImageOpenArgsOpenCVParams(BaseModel):
    """Parameters for OpenCV image loading."""

    dtype: str = Field(default="uint8", description="Data type of the image (e.g., 'uint8', 'float32')")
    flags: Literal[-1, 0, 1, 2, 4, 8, 16, 17, 32, 33, 64, 65, 128] = Field(
        default=-1, description="OpenCV flags for image loading (e.g., cv2.IMREAD_COLOR). Default is -1 for loading the image as is."
    )


class ImageOpenArgs(BaseModel):
    """Wrapper for specifying image format and associated parameters."""

    image_type: str = Field(default="", description="Image format (e.g., 'PNG', 'JPEG', 'RAW', and others). ")
    params: ImageOpenArgsRawPyParams | ImageOpenArgsRawParams | ImageOpenArgsOpenCVParams = Field(
        default_factory=ImageOpenArgsOpenCVParams, description="Parameters for the image"
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
