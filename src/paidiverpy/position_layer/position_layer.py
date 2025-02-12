"""Position layer module.

Process the images in the position layer.
"""

import logging
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.position_params import POSITION_LAYER_METHODS
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser


class PositionLayer(Paidiverpy):
    """Position layer class.

    This class processes the images in the position layer.

    Args:
        config_params (dict | ConfigParams, optional): The configuration parameters.
            It can contain the following keys / attributes:
            - input_path (str): The path to the input files.
            - output_path (str): The path to the output files.
            - metadata_path (str): The path to the metadata file.
            - metadata_type (str): The type of the metadata file.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of n_jobs.
        config_file_path (str): The path to the configuration file.
        config (Configuration): The configuration object.
        metadata (MetadataParser): The metadata object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        step_name (str): The name of the step.
        parameters (dict): The parameters for the step.
        config_index (int): The index of the configuration.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        parameters: dict | None = None,
        config_index: int | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        super().__init__(
            config_params=config_params,
            config_file_path=config_file_path,
            metadata=metadata,
            config=config,
            images=images,
            paidiverpy=paidiverpy,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )

        self.step_name = step_name
        if parameters:
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.layer_methods = POSITION_LAYER_METHODS
