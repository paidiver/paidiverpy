"""Position layer parameters models."""

from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class CalculateCornersParams(BaseModel):
    """Parameters for the overlapping resampling calculation."""

    omega: float = Field(0.5, description="Horizontal angle of view (in radians or normalized units)")
    theta: float = Field(0.5, description="Vertical angle of view (in radians or normalized units)")
    camera_distance: float = Field(1.12, description="Distance from camera to the scene (in meters)")
    raise_error: bool = Field(False, description="Raise error on failure")


POSITION_LAYER_METHODS = {
    "calculate_corners": {"params": CalculateCornersParams, "method": "calculate_corners"},
}

PositionParamsUnion = CalculateCornersParams | dict
