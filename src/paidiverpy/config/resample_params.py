"""Resample parameters dataclasses.

This module contains the dataclasses for the parameters of the convert layer
functions.
"""

from dataclasses import dataclass
from paidiverpy.utils.dynamic_classes import DynamicConfig


@dataclass
class ResampleDatetimeParams(DynamicConfig):
    """This class contains the parameters for the datetime resampling."""

    min: str = None
    max: str = None
    raise_error: bool = False


@dataclass
class ResampleDepthParams(DynamicConfig):
    """This class contains the parameters for the depth resampling."""

    by: str = "lower"
    value: float = None
    raise_error: bool = False


@dataclass
class ResampleAltitudeParams(DynamicConfig):
    """This class contains the parameters for the altitude resampling."""

    value: float = None
    raise_error: bool = False


@dataclass
class ResamplePitchRollParams(DynamicConfig):
    """This class contains the parameters for the pitch and roll res."""

    pitch: float = None
    roll: float = None
    raise_error: bool = False


@dataclass
class ResampleOverlappingParams(DynamicConfig):
    """This class contains the parameters for the overlapping resampling."""

    omega: float = 0.5
    theta: float = 0.5
    threshold: float = None
    camera_distance: float = 1.12
    raise_error: bool = False


@dataclass
class ResampleFixedParams(DynamicConfig):
    """This class contains the parameters for the fixed resampling."""

    value: int = 10
    raise_error: bool = False


@dataclass
class ResamplePercentParams(DynamicConfig):
    """This class contains the parameters for the percent resampling."""

    value: float = 0.1
    raise_error: bool = False


@dataclass
class ResampleRegionParams(DynamicConfig):
    """This class contains the parameters for the region resampling."""

    file: str = None
    limits: list[str] = None
    raise_error: bool = False


@dataclass
class ResampleObscureParams(DynamicConfig):
    """This class contains the parameters for the obscure resampling."""

    min: int = 0
    max: int = 1
    raise_error: bool = False


RESAMPLE_LAYER_METHODS = {
    "datetime": {"params": ResampleDatetimeParams, "method": "_by_datetime"},
    "depth": {"params": ResampleDepthParams, "method": "_by_depth"},
    "altitude": {"params": ResampleAltitudeParams, "method": "_by_altitude"},
    "pitch_roll": {"params": ResamplePitchRollParams, "method": "_by_pitch_roll"},
    "overlapping": {"params": ResampleOverlappingParams, "method": "_by_overlapping"},
    "fixed": {"params": ResampleFixedParams, "method": "_by_fixed_number"},
    "percent": {"params": ResamplePercentParams, "method": "_by_percent"},
    "region": {"params": ResampleRegionParams, "method": "_by_region"},
    "obscure": {"params": ResampleObscureParams, "method": "_by_obscure_photos"},
}
