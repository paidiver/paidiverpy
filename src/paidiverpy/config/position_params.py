"""Position layer parameters.

This module contains the dataclasses for the parameters of the convert layer
functions.
"""

from dataclasses import dataclass
from paidiverpy.utils import DynamicConfig


@dataclass
class ReprojectParams(DynamicConfig):
    """This class contains the parameters for the bit conversion."""

    placeholder: str = "placeholder"


POSITION_LAYER_METHODS = {
    "reproject": {"params": ReprojectParams, "method": "convert_bits"},
}
