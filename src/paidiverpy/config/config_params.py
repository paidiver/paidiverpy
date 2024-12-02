""" Configuration parameters module. """

from pathlib import Path
from typing import Dict, Optional
from paidiverpy.utils import DynamicConfig

REQUIRED_KEYS = ["input_path", "output_path", "metadata_path", "metadata_type", "track_changes", "n_jobs"]


class ConfigParams(DynamicConfig):
    """ Configuration parameters class.

    Args:
        config_params (Dict): The configuration parameters.
            It should have the following keys:
            - input_path (str): The input path.
            - output_path (str): The output path.
            - metadata_path (str): The metadata path.
            - metadata_type (str): The metadata type.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of jobs.

    Raises:
        ValueError: Invalid configuration parameters.
    """

    def __init__(self, config_params: Dict[str, Optional[str]]) -> None:

        self.config_params = self._validate_config_params(config_params)
        self.input_path = Path(self.config_params["input_path"])
        self.output_path = Path(self.config_params["output_path"])
        self.metadata_path = Path(self.config_params["metadata_path"])
        self.metadata_type = self.config_params["metadata_type"]
        self.track_changes = self.config_params["track_changes"]
        self.n_jobs = self.config_params["n_jobs"]


    def _validate_config_params(self, config_params: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        """Validate the configuration parameters.

        Args:
            config_params (Dict): The configuration parameters.
                It should have the following keys:
                - input_path (str): The input path.
                - output_path (str): The output path.
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

        missing_keys = dict_keys - elements_set

        if missing_keys:
            raise ValueError(f"Params {missing_keys} in config_params are not in required_keys.")

        for key in REQUIRED_KEYS:
            if key not in config_params:
                new_config_params[key] = None
            else:
                new_config_params[key] = config_params[key]
        return new_config_params
