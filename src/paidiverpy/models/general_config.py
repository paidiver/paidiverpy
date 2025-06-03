"""Configuration module."""

from pathlib import Path
from typing import ClassVar
from typing import Literal
from pydantic import Field
from pydantic import model_validator
from paidiverpy.models.client_params import ClientParams
from paidiverpy.models.open_params import ImageOpenArgs
from paidiverpy.models.step_config import ConvertConfig
from paidiverpy.models.step_config import SamplingConfig
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.data import PaidiverpyData
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.object_store import path_is_remote

logger = initialise_logging()


class GeneralConfig(BaseModel):
    """General configuration class.

    This class is used to define the general configuration from the configuration file
        or from the input from the user.

    """

    name: str = Field("raw", description="Name of the first step (the step to open images)")
    step_name: str = Field(
        "open", description="Step name. This is a placeholder for the first step name and should not be used in the configuration file."
    )
    sample_data: Literal["plankton_csv", "benthic_csv", "benthic_ifdo", "nef_raw", "benthic_raw_images"] | None = Field(
        None,
        description="Sample data to use for testing. If provided, it will override input_path, metadata_path, and metadata_type.",
    )
    input_path: str | Path | None = Field(None, description="Input path for image data. Can be a local path or a remote URL.")
    output_path: str | Path = Field("output", description="Output path for results. Can be a local path or a remote URL.")
    metadata_path: str | Path | None = Field(None, description="Path to metadata. Can be a local path or a remote URL.")
    metadata_type: (
        Literal[
            "IFDO",
            "CSV_FILE",
        ]
        | None
    ) = Field(None, description="Type of metadata. Can be 'IFDO' or 'CSV_FILE'")
    image_open_args: str | ImageOpenArgs = Field(
        "",
        description=(
            "Arguments to use when opening images. It can be a string with the image "
            "format or an ImageOpenArgs object. If it is a empty string, the type will be inferred from the file extension."
        ),
    )
    append_data_to_metadata: str | None = Field(
        None,
        description=("Path to append data to metadata. If provided, it will be used to append data to the metadata file."),
    )
    metadata_conventions: str | None = Field(
        None,
        description=("Metadata conventions to apply. If not provided, it will use the default conventions name described in the documentation."),
    )
    n_jobs: int = Field(1, description="Number of jobs for parallel processing")
    client: None | ClientParams = Field(default=None, description=("Dask Client configuration. If None, it will not use Dask Client."))

    track_changes: bool = Field(True, description="Whether to track config changes. If True, it will store in memory the output images on each step")
    rename: Literal["UUID", "datetime"] | None = Field(
        None, description="Field name to use for renaming. If not provided, the name will be the same as the input file name."
    )
    sampling: list[SamplingConfig] | None = Field(
        None,
        description=(
            "Sampling step configurations to be applied to the images before processing them. If not provided, no sampling will be applied."
        ),
    )

    convert: list[ConvertConfig] | None = Field(
        None,
        description=(
            "Convert step configurations to be applied to the images before processing them. If not provided, no conversion will be applied."
        ),
    )

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
