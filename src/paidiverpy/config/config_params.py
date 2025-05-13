"""Configuration parameters module."""

from pathlib import Path
from paidiverpy.utils.dynamic_classes import DynamicConfig

REQUIRED_KEYS = ["input_path", "output_path", "metadata_path", "metadata_type", "image_open_args"]


class ConfigParams(DynamicConfig):
    """Configuration parameters class.

    Args:
        config_params (Dict): The configuration parameters.
            It should have the following keys:
            - input_path (str): The input path.
            - output_path (str): The output path.
            - image_open_args (str): The image type.
            - metadata_path (str): The metadata path.
            - metadata_type (str): The metadata type.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of jobs.

    Raises:
        ValueError: Invalid configuration parameters.
    """

    def __init__(self, config_params: dict[str, str | None]) -> None:
        self.config_params = self._validate_config_params(config_params)
        self.input_path = Path(self.config_params["input_path"])
        self.output_path = Path(self.config_params["output_path"])
        self.image_open_args = self.config_params["image_open_args"]
        self.metadata_path = Path(self.config_params["metadata_path"])
        self.metadata_type = self.config_params["metadata_type"]
        self.track_changes = self.config_params.get("track_changes", True)
        self.n_jobs = self.config_params.get("n_jobs", 1)

    def _validate_config_params(self, config_params: dict[str, str | None]) -> dict[str, str | None]:
        """Validate the configuration parameters.

        Args:
            config_params (Dict): The configuration parameters.
                It should have the following keys:
                - input_path (str): The input path.
                - output_path (str): The output path.
                - image_open_args (str): The image type.
                - metadata_path (str): The metadata path.
                - metadata_type (str): The metadata type.
                - track_changes (bool): Whether to track changes.
                - n_jobs (int): The number of jobs.

        Raises:
            ValueError: Invalid configuration parameters.

        Returns:
            Dict: The validated configuration parameters.
        """
        new_config_params = {}

        dict_keys = set(config_params.keys())
        elements_set = set(REQUIRED_KEYS)

        missing_keys = elements_set - dict_keys

        if missing_keys:
            msg = f"Error in config_params: params {missing_keys} are missing."
            raise ValueError(msg)

        for key in REQUIRED_KEYS:
            new_config_params[key] = config_params.get(key)
        return new_config_params
