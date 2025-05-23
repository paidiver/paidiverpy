"""Configuration parameters module."""

from pathlib import Path
from pydantic import model_validator
from paidiverpy.utils.base_model import BaseModel


class ConfigParams(BaseModel):
    """Configuration parameters using Pydantic.

    Fields:
        input_path (Path): The input path.
        output_path (Path): The output path.
        image_open_args (str): The image type.
        metadata_path (Path): The metadata path.
        metadata_type (str): The metadata type.
        track_changes (bool): Whether to track changes. Defaults to True.
        n_jobs (int): Number of jobs. Defaults to 1.
    """

    input_path: Path
    output_path: Path
    image_open_args: str
    metadata_path: Path
    metadata_type: str
    track_changes: bool = True
    n_jobs: int = 1

    @model_validator(mode="before")
    @classmethod
    def validate_required_keys(cls, values: dict) -> dict:
        """Validate the required keys in the configuration parameters.

        Args:
            values (dict): The values to validate.

        Raises:
            ValueError: If any of the required keys are missing.

        Returns:
            dict: The validated values.
        """
        required_keys = ["input_path", "output_path", "metadata_path", "metadata_type", "image_open_args"]
        missing = [key for key in required_keys if key not in values]
        if missing:
            msg = f"Error in config_params: params {missing} are missing."
            raise ValueError(msg)
        return values
