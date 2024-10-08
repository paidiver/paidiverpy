"""Pipeline parameters.

Pipeline parameters for the paidiverpy package
"""

from paidiverpy.color_layer import ColorLayer
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.open_layer import OpenLayer
from paidiverpy.position_layer import PositionLayer
from paidiverpy.resample_layer import ResampleLayer

STEPS_CLASS_TYPES = {
    "position": PositionLayer,
    "sampling": ResampleLayer,
    "convert": ConvertLayer,
    "color": ColorLayer,
    "raw": OpenLayer,
}
