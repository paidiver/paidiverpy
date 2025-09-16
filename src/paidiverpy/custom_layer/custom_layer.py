"""Custom layer module.

This module contains the CustomLayer class for processing the images in the
custom layer.
"""

import logging
from typing import Any
from dask.distributed import Client
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.custom_params import CustomParams
from paidiverpy.utils.install_packages import check_and_install_dependencies


class CustomLayer(Paidiverpy):
    """CustomLayer class.

    Process the images in the custom layer.

    Args:
        parameters (dict): The parameters for the step.
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
        client (Client): The Dask client.
        config_index (int): The index of the configuration.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        config_params: dict[str, Any] | ConfigParams | None = None,
        config_file_path: str | None = None,
        config: Configuration | None = None,
        metadata: MetadataParser | None = None,
        images: ImagesLayer | None = None,
        paidiverpy: Paidiverpy | None = None,
        step_name: str | None = None,
        client: Client | None = None,
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
            client=client,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )

        self.step_name = step_name

        self.config_index = self.config.add_step(config_index, parameters, step_class=CustomLayer)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.raise_error = self._calculate_raise_error()

    def run(self) -> None:
        """Custom Layer run method.

        Run the custom layer steps on the images based on the configuration
        file or parameters.

        Args:
            add_new_step (bool, optional): Whether to add a new step to the images object.
        Defaults to True.
        """
        check_and_install_dependencies(self.step_metadata.get("dependencies"), self.step_metadata.get("dependencies_path"))
        test = self.step_metadata.get("test", False)
        algorithm_name = self.step_metadata.get("name", "")
        try:
            method = getattr(self, algorithm_name)
        except AttributeError as e:
            msg = f"Method {algorithm_name} not found in CustomLayer."
            self.logger.error(msg)
            if self.raise_error:
                raise AttributeError(msg) from e
            return
        params = self.step_metadata.get("params") or {}
        params = CustomParams(**params) if isinstance(params, dict) else params
        processing_type = self.step_metadata.get("processing_type")
        if processing_type == "dataset":
            images = self.images.get_step(step=len(self.images.images) - 1)
            images = self.process_dataset(images, method, params)
        else:
            images = self.process_images(method, params)
        if not test:
            self.step_name = f"step_{self.config_index}" if not self.step_name else self.step_name
            self.images.add_step(
                step=self.step_name,
                images=images,
                step_metadata=self.step_metadata,
                track_changes=self.track_changes,
            )
