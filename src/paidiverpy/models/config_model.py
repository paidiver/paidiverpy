"""Configuration model for Paidiverpy."""

from pydantic import Field
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.models.step_config import StepConfigUnion
from paidiverpy.utils.base_model import BaseModel


class ConfigModel(BaseModel):
    """Step configuration model."""

    general: GeneralConfig = Field(description="General configuration")
    steps: list[StepConfigUnion] | None = Field(default=None, description="List of step configurations")
