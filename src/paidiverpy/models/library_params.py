"""Configuration models for functions provided by image libraries."""

from typing import Any
from typing import Literal
from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class LibraryConfig(BaseModel):
    """Configuration for applying an OpenCV or scikit-image function."""

    name: str | None = Field("library", description="Name of the step")
    step_name: str | None = Field(None, description="Step name")
    test: bool = Field(False, description="Test mode")
    library: Literal["opencv", "skimage"] = Field(description="Library containing the function")
    function: str = Field(description="Public dotted path to the function within the selected library")
    image_param: str | None = Field(
        None,
        description="Keyword parameter that receives the image; by default the image is the first positional argument",
    )
    result_index: int | None = Field(
        None,
        description="Index of the image in a tuple or list returned by the function",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Keyword parameters passed to the function")
