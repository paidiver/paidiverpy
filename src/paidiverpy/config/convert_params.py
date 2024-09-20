""" This module contains the dataclasses for the parameters of the convert layer functions. """

from dataclasses import dataclass
from utils import DynamicConfig


@dataclass
class BitParams(DynamicConfig):
    """This class contains the parameters for the bit conversion."""

    output_bits: int = 8


@dataclass
class ToParams(DynamicConfig):
    """This class contains the parameters for the channel conversion."""

    to: str = "uint8"
    channel_selector: int = 0


@dataclass
class BayerPatternParams(DynamicConfig):
    """This class contains the parameters for the Bayer pattern conversion"""
    def __init__(self) -> None:
        self.bayer_pattern: str = "BGGR"

@dataclass
class NormalizeParams(DynamicConfig):
    """This class contains the parameters for the image normalization."""

    min: float = 0
    max: float = 1


@dataclass
class ResizeParams(DynamicConfig):
    """This class contains the parameters for the image resizing."""

    min: int = 256
    max: int = 256


@dataclass
class CropParams(DynamicConfig):
    """This class contains the parameters for the image cropping."""

    x: tuple = (0, -1)
    y: tuple = (0, -1)


CONVERT_LAYER_METHODS = {
    "bits": {"params": BitParams, "method": "convert_bits"},
    "to": {"params": ToParams, "method": "channel_convert"},
    "bayer_pattern": {"params": BayerPatternParams, "method": "get_bayer_pattern"},
    "normalize": {"params": NormalizeParams, "method": "normalize_image"},
    "resize": {"params": ResizeParams, "method": "resize"},
    "crop": {"params": CropParams, "method": "crop_images"},
}
