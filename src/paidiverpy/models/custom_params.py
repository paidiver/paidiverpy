"""Custom parameters dataclasses.

This module contains the dataclasses for the parameters used in the custom_params module.
"""

from typing import ClassVar
from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class CustomParams(BaseModel):
    """Parameters for the custom_params module with support for arbitrary fields."""

    raise_error: bool = Field(default=False, description="Raise error on failure")

    model_config: ClassVar[dict] = {
        "extra": "allow",
    }


# CustomParamsUnion = CustomParams | dict
