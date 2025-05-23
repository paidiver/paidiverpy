"""Configuration module."""

import logging
from pathlib import Path
from typing import Any
from typing import ClassVar
from pydantic import Field
from pydantic import model_validator
from paidiverpy.config.client_params import ClientParams
from paidiverpy.config.step_config import ConvertConfig
from paidiverpy.config.step_config import SamplingConfig
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.data import PaidiverpyData
from paidiverpy.utils.object_store import path_is_remote

logger = logging.getLogger(__name__)


class GeneralConfig(BaseModel):
    """General configuration class.

    This class is used to define the general configuration from the configuration file
        or from the input from the user.

    """

    name: str = Field("raw", description="Name of the configuration")
    step_name: str = Field("open", description="Step name")
    sample_data: str | None = Field(None, description="Sample data type")
    input_path: str | Path | None = Field(None, description="Input path for image data")
    output_path: str | Path = Field("output", description="Output path for results")
    metadata_path: str | Path | None = Field(None, description="Path to metadata")
    metadata_type: str | None = Field(None, description="Type of metadata")
    image_open_args: Any = Field(None, description="Arguments to use when opening images")
    append_data_to_metadata: str | None = Field(None, description="Append data to metadata")
    metadata_conventions: str | None = Field(None, description="Metadata conventions to apply")
    n_jobs: int = Field(1, description="Number of jobs for parallel processing")
    client: None | ClientParams = Field(None, description="Dask client object if available")
    track_changes: bool = Field(True, description="Whether to track config changes")
    rename: str | None = Field(None, description="Field name to use for renaming")
    sampling: list[SamplingConfig] | None = Field(None, description="Sampling step configurations")
    convert: list[ConvertConfig] | None = Field(None, description="Convert step configurations")

    model_config: ClassVar[dict] = {
        "frozen": False,
        "json_schema_extra": {
            "anyOf": [
                {"required": ["input_path", "metadata_path", "metadata_type"], "not": {"required": ["sample_data"]}},
                {"required": ["sample_data"]},
            ]
        },
    }

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, values: dict) -> dict:
        """Validate the fields of the configuration.

        Args:
            values (dict): The values to validate.

        Returns:
            dict: The validated values.
        """
        sample_data = values.get("sample_data")
        if sample_data:
            data = PaidiverpyData()
            information = data.load(sample_data)
            values["input_path"] = Path(information["input_path"])
            values["metadata_path"] = Path(information["metadata_path"])
            values["metadata_type"] = information["metadata_type"]
            values["image_open_args"] = information["image_open_args"]
            values["append_data_to_metadata"] = information.get("append_data_to_metadata")
        else:
            input_path = values.get("input_path")
            if input_path:
                values["input_path"] = Path(str(input_path)) if not path_is_remote(input_path) else input_path

        output_path = values.get("output_path")
        if output_path:
            values["output_path"] = Path(str(output_path)) if not path_is_remote(output_path) else output_path

        # Convert step configurations into StepConfig instances
        for step_type in ["sampling", "convert"]:
            steps = values.get(step_type)
            if steps:
                for step in steps:
                    step["step_name"] = step_type
                    step["name"] = step_type
                if step_type == "sampling":
                    values[step_type] = [SamplingConfig(**step) for step in steps]
                else:
                    values[step_type] = [ConvertConfig(**step) for step in steps]

        return values

    @model_validator(mode="after")
    def check_required_fields(self) -> "GeneralConfig":
        """Ensure output_path is provided and either sample_data or input_path is set."""
        if not self.output_path:
            msg = "'output_path' is required."
            raise ValueError(msg)

        if not self.sample_data and not self.input_path:
            msg = "Either 'sample_data' or 'input_path' must be provided."
            raise ValueError(msg)

        return self

    def update(self, **updates: dict) -> "GeneralConfig":
        """Update the model in-place with new values."""
        for key, value in updates.items():
            setattr(self, key, value)
        validated = self.__class__.model_validate(self.model_dump())
        for key, val in validated.model_dump().items():
            setattr(self, key, val)
        return self
