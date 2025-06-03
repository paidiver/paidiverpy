"""Sampling parameters dataclasses.

This module contains the dataclasses for the parameters of the convert layer
functions.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class SamplingDatetimeParams(BaseModel):
    """Parameters for datetime resampling."""

    min: str | None = Field(
        default=None,
        description=("Minimum datetime bound (ISO 8601 format). If not provided, it will use the earliest datetime in the dataset."),
    )
    max: str | None = Field(
        default=None,
        description=("Maximum datetime bound (ISO 8601 format). If not provided, it will use the latest datetime in the dataset."),
    )
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingDepthParams(BaseModel):
    """Parameters for depth resampling."""

    by: Literal["lower", "upper"] = Field(default="lower", description="Resampling strategy (e.g., 'lower', 'upper')")
    value: float | None = Field(default=None, description="Depth value threshold")
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingAltitudeParams(BaseModel):
    """Parameters for altitude resampling."""

    by: Literal["lower", "upper"] = Field(default="lower", description="Resampling strategy (e.g., 'lower', 'upper')")
    value: float | None = Field(default=None, description="Altitude value threshold")
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingPitchRollParams(BaseModel):
    """Parameters for pitch and roll resampling."""

    pitch: float = Field(default=9999, description="Pitch tolerance value")
    roll: float = Field(default=9999, description="Roll tolerance value")
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingOverlappingParams(BaseModel):
    """Parameters for overlapping resampling."""

    omega: float = Field(default=0.5, description="Horizontal angle of view")
    theta: float = Field(default=0.5, description="Vertical angle of view")
    threshold: float = Field(default=1, description="Overlap threshold in percentage")
    camera_distance: float = Field(default=1.12, description="Distance from camera to subject")
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingFixedParams(BaseModel):
    """Parameters for fixed interval resampling."""

    value: int = Field(default=10, description="Fixed interval value")
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingPercentParams(BaseModel):
    """Parameters for percent-based resampling."""

    value: float = Field(default=0.1, description="Percent value to resample")
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingRegionParams(BaseModel):
    """Parameters for region-based resampling."""

    file: str | None | Path = Field(default=None, description="Path to the region file")
    limits: dict | None = Field(
        default=None,
        description=("Dict of region limits. Format {'min_lon': value, 'max_lon': value, 'min_lat': value, 'max_lat': value}"),
    )
    raise_error: bool = Field(default=False, description="Raise error on failure")


class SamplingObscureParams(BaseModel):
    """Parameters for obscure-based resampling."""

    min: float = Field(default=0, description="Minimum threshold for obscuring")
    max: float = Field(default=1, description="Maximum threshold for obscuring")
    channel: str = Field(default="mean", description="Channel selection strategy")
    raise_error: bool = Field(default=False, description="Raise error on failure")


SAMPLING_LAYER_METHODS = {
    "datetime": {"params": SamplingDatetimeParams, "method": "_by_datetime"},
    "depth": {"params": SamplingDepthParams, "method": "_by_depth"},
    "altitude": {"params": SamplingAltitudeParams, "method": "_by_altitude"},
    "pitch_roll": {"params": SamplingPitchRollParams, "method": "_by_pitch_roll"},
    "overlapping": {"params": SamplingOverlappingParams, "method": "_by_overlapping"},
    "fixed": {"params": SamplingFixedParams, "method": "_by_fixed_number"},
    "percent": {"params": SamplingPercentParams, "method": "_by_percent"},
    "region": {"params": SamplingRegionParams, "method": "_by_region"},
    "obscure": {"params": SamplingObscureParams, "method": "_by_obscure_images"},
}

SamplingParamsUnion = (
    SamplingDatetimeParams
    | SamplingDepthParams
    | SamplingAltitudeParams
    | SamplingPitchRollParams
    | SamplingOverlappingParams
    | SamplingFixedParams
    | SamplingPercentParams
    | SamplingRegionParams
    | SamplingObscureParams
    | dict
)
