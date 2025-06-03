"""Pipeline parameters.

Pipeline parameters for the paidiverpy package
"""

from paidiverpy.colour_layer import ColourLayer
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.custom_layer import CustomLayer
from paidiverpy.open_layer import OpenLayer
from paidiverpy.position_layer import PositionLayer
from paidiverpy.sampling_layer import SamplingLayer

STEPS_CLASS_TYPES = {
    "position": PositionLayer,
    "sampling": SamplingLayer,
    "convert": ConvertLayer,
    "colour": ColourLayer,
    "raw": OpenLayer,
    "custom": CustomLayer,
}
