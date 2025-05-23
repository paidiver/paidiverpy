"""Step configuration module."""

import logging
from typing import Any
from typing import ClassVar
from pydantic import Field
from pydantic import model_validator
from paidiverpy.config.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.config.colour_params import ColourParamsUnion
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.convert_params import ConvertParamsUnion
from paidiverpy.config.custom_params import CustomParams
from paidiverpy.config.custom_params import CustomParamsUnion
from paidiverpy.config.position_params import POSITION_LAYER_METHODS
from paidiverpy.config.position_params import PositionParamsUnion
from paidiverpy.config.sampling_params import SAMPLING_LAYER_METHODS
from paidiverpy.config.sampling_params import SamplingParamsUnion
from paidiverpy.utils.base_model import BaseModel

steps_params_mapping = {
    "colour": COLOUR_LAYER_METHODS,
    "convert": CONVERT_LAYER_METHODS,
    "position": POSITION_LAYER_METHODS,
    "sampling": SAMPLING_LAYER_METHODS,
}

logger = logging.getLogger(__name__)


class StepConfig(BaseModel):
    """Step configuration model."""

    name: str | None = Field(None, description="Name of the step")
    step_name: str | None = Field(None, description="Step name")
    test: bool = Field(False, description="Test mode")
    file_path: str | None = Field(None, description="File path for custom step")
    class_name: str | None = Field(None, description="Class name for custom step")
    mode: str | None = Field(None, description="Mode for the step")
    params: Any = Field(default_factory=dict, description="Parameters for the step")

    model_config: ClassVar[dict] = {
        "frozen": False,
    }

    @model_validator(mode="before")
    @classmethod
    def resolve_params_schema(cls, values: dict) -> dict:
        """Resolve the parameters schema based on the step name and mode.

        Args:
            values (dict): The values to validate.

        Returns:
            dict: The validated values.
        """
        if isinstance(values, StepConfig):
            return values
        step_name = values.get("step_name")
        params = values.get("params", {})
        mode = values.get("mode")

        if step_name == "custom":
            values["params"] = CustomParams(**params)
        else:
            if step_name not in steps_params_mapping:
                msg = f"Unknown step_name: '{step_name}'"
                raise ValueError(msg)
            if mode is None:
                msg = "Missing 'mode' for non-custom step"
                raise ValueError(msg)
            method_dict = steps_params_mapping[step_name]
            if mode not in method_dict:
                msg = f"Mode '{mode}' not valid for step '{step_name}'"
                raise ValueError(msg)
            param_class = method_dict[mode]["params"]
            values["params"] = param_class(**params)
        return values

    def update(self, **updates: dict) -> "StepConfig":
        """Update the model in-place with new values."""
        for key, value in updates.items():
            setattr(self, key, value)
        validated = self.__class__.model_validate(self.model_dump())
        for key, val in validated.model_dump().items():
            setattr(self, key, val)
        return self


class PositionConfig(StepConfig):
    """Position configuration model."""

    params: PositionParamsUnion | None = Field(default=None, description="Position parameters")


class ColourConfig(StepConfig):
    """Colour configuration model."""

    params: ColourParamsUnion | None = Field(default=None, description="Colour parameters")


class ConvertConfig(StepConfig):
    """Convert configuration model."""

    params: ConvertParamsUnion | None = Field(default=None, description="Convert parameters")


class SamplingConfig(StepConfig):
    """Sampling configuration model."""

    params: SamplingParamsUnion | None = Field(default=None, description="Sampling parameters")


class CustomConfig(StepConfig):
    """Custom configuration model."""

    params: CustomParamsUnion | None = Field(default=None, description="Custom parameters")


StepConfigUnion = PositionConfig | ColourConfig | ConvertConfig | SamplingConfig | CustomConfig | SamplingConfig
