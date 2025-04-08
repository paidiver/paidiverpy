"""Position layer parameters.

This module contains the dataclasses for the parameters of the convert layer
functions.
"""

from dataclasses import dataclass
from paidiverpy.utils.dynamic_classes import DynamicConfig


@dataclass
class CalculateCornersParams(DynamicConfig):
    """This class contains the parameters for the overlapping resampling."""

    omega: float = 0.5
    theta: float = 0.5
    camera_distance: float = 1.12
    raise_error: bool = False


POSITION_LAYER_METHODS = {
    "calculate_corners": {"params": CalculateCornersParams, "method": "calculate_corners"},
}
