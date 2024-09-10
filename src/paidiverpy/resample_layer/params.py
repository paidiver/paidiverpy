""" This module contains the dataclasses for the parameters of the convert layer functions. """

from dataclasses import dataclass
from typing import List


@dataclass
class ResampleDatetimeParams:
    """This class contains the parameters for the datetime resampling."""

    min: str = None
    max: str = None


@dataclass
class ResampleDepthParams:
    """This class contains the parameters for the depth resampling."""

    by: str = "lower"
    value: float = None


@dataclass
class ResampleAltitudeParams:
    """This class contains the parameters for the altitude resampling."""

    value: float = None


@dataclass
class ResamplePitchRollParams:
    """This class contains the parameters for the pitch and roll res"""

    pitch: float = None
    roll: float = None


@dataclass
class ResampleOverlappingParams:
    """This class contains the parameters for the overlapping resampling."""

    omega: float = 0.5
    theta: float = 0.5
    threshold: float = 0.5


@dataclass
class ResampleFixedParams:
    """This class contains the parameters for the fixed resampling"""

    value: int = 10


@dataclass
class ResamplePercentParams:
    value: float = 0.1


@dataclass
class ResampleRegionParams:
    """This class contains the parameters for the region resampling"""

    file: str = None
    limits: List[str] = None


@dataclass
class ResampleObscureParams:
    """This class contains the parameters for the obscure resampling"""

    min: int = 0
    max: int = 1


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
