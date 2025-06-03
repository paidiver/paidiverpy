"""Convert layer parameters models."""

from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class BitParams(BaseModel):
    """This class contains the parameters for the bit conversion."""

    output_bits: int = Field(8, description="Number of bits in output image")
    raise_error: bool = Field(False, description="Raise error on failure")


class ToParams(BaseModel):
    """This class contains the parameters for the channel conversion."""

    to: str = Field("uint8", description="Target data type")
    channel_selector: int = Field(0, description="Index of channel to select")
    raise_error: bool = Field(False, description="Raise error on failure")


class NormalizeParams(BaseModel):
    """This class contains the parameters for the image normalization."""

    min: float = Field(0, description="Minimum normalization value")
    max: float = Field(1, description="Maximum normalization value")
    method: str = Field("minmax", description="Normalization method")
    raise_error: bool = Field(False, description="Raise error on failure")


class ResizeParams(BaseModel):
    """This class contains the parameters for the image resizing."""

    size: dict | None = Field(None, description="Target size {'width': value, 'height': value}")
    preserve_aspect: bool = Field(True, description="Preserve aspect ratio")
    scale: float | dict = Field(1, description="Scale factor. Format {'width': value, 'height': value} or float for uniform scaling")
    interpolation: str = Field("linear", description="Interpolation method")
    raise_error: bool = Field(False, description="Raise error on failure")


class CropParams(BaseModel):
    """This class contains the parameters for the image cropping."""

    size: dict | float = Field(1, description="Crop size. Format {'width': value, 'height': value} or float for percentage")
    size_type: str = Field("percent", description="Size type: percent or pixels")
    mode: str = Field("center", description="Crop mode")
    top_left: dict = Field({"top": 0, "left": 0}, description="Top-left corner for cropping. Format {'top': value, 'left': value}")
    raise_error: bool = Field(False, description="Raise error on failure")


CONVERT_LAYER_METHODS = {
    "bits": {"params": BitParams, "method": "convert_bits"},
    "to": {"params": ToParams, "method": "channel_convert"},
    "normalize": {"params": NormalizeParams, "method": "normalize_image"},
    "resize": {"params": ResizeParams, "method": "resize"},
    "crop": {"params": CropParams, "method": "crop_images"},
}

ConvertParamsUnion = BitParams | ToParams | NormalizeParams | ResizeParams | CropParams | dict
