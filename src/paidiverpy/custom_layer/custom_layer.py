"""Color layer module.

This module contains the ColorLayer class for processing the images in the
color layer.
"""

import importlib.util
import logging
from importlib.resources import files
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.custom_params import CustomParams
from paidiverpy.utils.docker import is_running_in_docker
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
        config_index (int): The index of the configuration.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        parameters: dict,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
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
        algorithm_name = self.step_metadata.get("name")
        file_path = self.step_metadata.get("file_path")
        is_docker = is_running_in_docker()
        if is_docker:
            file_name = file_path.split("/")[-1]
            file_path = "/app/custom_algorithms/" + file_name
        if self.step_metadata.get("file_path") == "example":
            file_path = files("paidiverpy").joinpath("custom_layer/_custom_algorithm_example.py")
        elif self.step_metadata.get("file_path") == "example_dataset":
            file_path = files("paidiverpy").joinpath("custom_layer/_custom_algorithm_example_dataset.py")
        class_name = self.step_metadata.get("class_name")
        check_and_install_dependencies(self.step_metadata.get("dependencies"), self.step_metadata.get("dependencies_path"))
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        params = CustomParams(**params) if isinstance(params, dict) else params
        method = self.load_custom_algorithm(file_path, class_name, algorithm_name)
        images = self.images.get_step(step=len(self.images.images) - 1)
        processing_type = self.step_metadata.get("processing_type")
        if processing_type == "dataset":
            image_list, metadata = self.process_dataset(images, method, params, custom=True)
        else:
            image_list, metadata = (
                self.process_sequentially(images, method, params, custom=True)
                if self.n_jobs == 1
                else self.process_parallel(images, method, params, custom=True)
            )
        if not test:
            self.step_name = algorithm_name if not self.step_name else self.step_name
            self.set_metadata(metadata, flag=len(self.images.images))
            self.images.add_step(
                step=self.step_name,
                images=image_list,
                step_metadata=self.step_metadata,
                metadata=self.get_metadata(),
                track_changes=self.track_changes,
            )

    def load_custom_algorithm(self, file_path: str, class_name: str, algorithm_name: str) -> callable:
        """Load a custom algorithm class.

        Args:
            file_path (str): The file path of the custom algorithm.
            class_name (str): The class name.
            algorithm_name (str): The algorithm name.

        Returns:
            class: The custom algorithm class.
        """
        spec = importlib.util.spec_from_file_location(algorithm_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, class_name)
