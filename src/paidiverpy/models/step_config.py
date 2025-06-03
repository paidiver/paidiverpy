"""Step configuration module."""

from typing import Any
from typing import ClassVar
from typing import Literal
from typing import cast
from pydantic import Field
from pydantic import model_validator
from paidiverpy.models.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.models.colour_params import ColourParamsUnion
from paidiverpy.models.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.models.convert_params import ConvertParamsUnion
from paidiverpy.models.custom_params import CustomParams
from paidiverpy.models.position_params import POSITION_LAYER_METHODS
from paidiverpy.models.position_params import PositionParamsUnion
from paidiverpy.models.sampling_params import SAMPLING_LAYER_METHODS
from paidiverpy.models.sampling_params import SamplingParamsUnion
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.logging_functions import initialise_logging

# from paidiverpy.config.custom_params import CustomParamsUnion

steps_params_mapping = {
    "colour": COLOUR_LAYER_METHODS,
    "convert": CONVERT_LAYER_METHODS,
    "position": POSITION_LAYER_METHODS,
    "sampling": SAMPLING_LAYER_METHODS,
}

PositionModeLiteral = cast(type, Literal.__getitem__(tuple(POSITION_LAYER_METHODS.keys())))
ColourModeLiteral = cast(type, Literal.__getitem__(tuple(COLOUR_LAYER_METHODS.keys())))
ConvertModeLiteral = cast(type, Literal.__getitem__(tuple(CONVERT_LAYER_METHODS.keys())))
SamplingModeLiteral = cast(type, Literal.__getitem__(tuple(SAMPLING_LAYER_METHODS.keys())))

logger = initialise_logging()


class StepConfig(BaseModel):
    """Step configuration model."""

    name: str | None = Field(None, description="Name of the step")
    step_name: str | None = Field(None, description="Step name")
    test: bool = Field(False, description="Test mode")
    params: Any = Field(default_factory=dict, description="Parameters for the step")
    mode: PositionModeLiteral | ColourModeLiteral | ConvertModeLiteral | SamplingModeLiteral = Field(
        description="Mode for the position step",
    )
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

    name: str | None = Field("position", description="Name of the step")
    mode: PositionModeLiteral = Field(
        description="Mode for the position step",
    )
    test: bool = Field(False, description="Test mode")
    params: PositionParamsUnion | None = Field(default=None, description="Position parameters")
    processing_type: Literal["image", "dataset"] = Field(
        "image",
        description=("If the images are processed individually (image option) or as a dataset (dataset option)"),
    )


class ColourConfig(StepConfig):
    """Colour configuration model."""

    name: str | None = Field("colour", description="Name of the step")
    mode: ColourModeLiteral = Field(
        description="Mode for the colour step",
    )
    test: bool = Field(False, description="Test mode")
    params: ColourParamsUnion | None = Field(default=None, description="Colour parameters")
    processing_type: Literal["image", "dataset"] = Field(
        "image",
        description=("If the images are processed individually (image option) or as a dataset (dataset option)"),
    )


class ConvertConfig(StepConfig):
    """Convert configuration model."""

    name: str | None = Field("convert", description="Name of the step")
    mode: ConvertModeLiteral = Field(description="Mode for the convert step")
    test: bool = Field(False, description="Test mode")
    params: ConvertParamsUnion | None = Field(default=None, description="Convert parameters")
    processing_type: Literal["image", "dataset"] = Field(
        "image",
        description=("If the images are processed individually (image option) or as a dataset (dataset option)"),
    )


class SamplingConfig(StepConfig):
    """Sampling configuration model."""

    name: str | None = Field("sampling", description="Name of the step")
    mode: SamplingModeLiteral = Field(description="Mode for the sampling step")
    test: bool = Field(False, description="Test mode")
    params: SamplingParamsUnion | None = Field(default=None, description="Sampling parameters")
    processing_type: Literal["image", "dataset"] = Field(
        "image",
        description=("If the images are processed individually (image option) or as a dataset (dataset option)"),
    )


class CustomConfig(BaseModel):
    """Custom configuration model."""

    name: str | None = Field("custom", description="Name of the step")
    file_path: str = Field(description="File path for custom step")
    class_name: str = Field(description="Class name for custom step")
    step_name: str | None = Field(None, description="Step name")
    test: bool = Field(False, description="Test mode")
    processing_type: Literal["image", "dataset"] = Field(
        "image",
        description=("If the images are processed individually (image option) or as a dataset (dataset option)"),
    )
    dependencies: str | None = Field(
        None,
        description=("Dependencies for the custom step. It should be a string with each dependency separated by commas."),
    )
    dependencies_path: str | None = Field(None, description="Path to the requirements file for the custom step")
    params: CustomParams | None = Field(default=None, description="Custom parameters")


StepConfigUnion = ColourConfig | ConvertConfig | CustomConfig | PositionConfig | SamplingConfig
