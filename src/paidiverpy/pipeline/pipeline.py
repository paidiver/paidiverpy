"""Pipeline builder class for image preprocessing."""

import gc
import logging
from jsonschema import ValidationError
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.open_layer import OpenLayer
from paidiverpy.pipeline.pipeline_params import STEPS_CLASS_TYPES
from paidiverpy.utils import formating_html

STEP_WITHOUT_PARAMS = 2
STEP_WITH_PARAMS = 3


class Pipeline(Paidiverpy):
    """Pipeline builder class for image preprocessing.

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
        steps (list[tuple], optional): The steps of the pipeline.
        track_changes (bool): Whether to track changes. Defaults to None, which means
            it will be set to the value of the configuration file.
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
        steps: list[tuple] | None = None,
        track_changes: bool | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        super().__init__(
            config_params=config_params,
            config_file_path=config_file_path,
            metadata=metadata,
            config=config,
            track_changes=track_changes,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )

        if steps is None:
            steps = self._convert_config_to_steps()
        else:
            for step in steps:
                step_name = self._get_step_name(step[1])
                name = step[0]
                step[2]["name"] = name
                step[2]["step_name"] = step_name
                if name == "raw":
                    self.config.add_general(step[2])
                else:
                    self.config.add_step(parameters=step[2], step_class=step[1])
        self.steps = steps
        self.runned_steps = -1

    def run(self, from_step: int | None = None, close_client: bool = True) -> None:
        """Run the pipeline.

        Args:
            from_step (int, optional): The step to start from. Defaults to None,
                which means the pipeline will start from the last runned step.
            close_client (bool, optional): Whether to close the client. Defaults to True.

        Raises:
            ValueError: No steps defined for the pipeline
            ValueError: Invalid step format
        """
        self._validate_pipeline()
        self._validate_from_step(from_step)

        if not self.client:
            self.logger.info("Processing images using %s cores", self.n_jobs)
        else:
            self.logger.info("Processing images using Dask client using the following dashboard link: %s", self.client.dashboard_link)
        for index, step in enumerate(self.steps):
            if index > self.runned_steps:
                step_name, step_class, step_params = self._get_steps_params(step)
                self.logger.info(
                    "Running step %s: %s - %s",
                    index,
                    step_name,
                    step_class.__name__,
                )
                step_params["step_name"] = self._get_step_name(step_class)
                step_params["name"] = step_name
                if step_name == "raw":
                    step_instance = step_class(
                        paidiverpy=self,
                        step_name=step_name,
                        parameters=step_params,
                    )
                    self.metadata.dataset_metadata["input_path"] = str(self.config.general.input_path)
                else:
                    step_instance = step_class(
                        paidiverpy=self,
                        step_name=step_name,
                        parameters=step_params,
                        config_index=index - 1,
                    )
                step_instance.run()
                if not step_params.get("test", False):
                    self.images = step_instance.images
                    self.set_metadata(step_instance.get_metadata(flag="all"))
                    self.runned_steps = index
                self.logger.info("Step %s completed", index)

                del step_instance
                gc.collect()
        if self.client and close_client:
            self.client.close()

    def _validate_pipeline(self) -> None:
        """Validate the pipeline.

        Raises:
            ValueError: No steps defined for the pipeline
        """
        if not self.steps:
            self.logger.error("No steps defined for the pipeline")
            msg = "No steps defined for the pipeline"
            raise ValueError(msg)

    def _validate_from_step(self, from_step: int | None) -> None:
        """Validate the from_step parameter."""
        if from_step is not None:
            if len(self.images.images) > from_step:
                self.runned_steps = from_step
                self.clear_steps(from_step + 1)
            else:
                self.logger.warning(
                    "Step %s does not exist. Run the pipeline from the beginning",
                    from_step,
                )

    def _get_steps_params(self, step: tuple) -> tuple:
        """Get the parameters of the step.

        Args:
            step (tuple): The step.
        """
        if len(step) == STEP_WITHOUT_PARAMS:
            step_name, step_class = step
            step_params = {}
        elif len(step) == STEP_WITH_PARAMS:
            step_name, step_class, step_params = step
        return step_name, step_class, step_params

    def export_config(self, output_path: str | None = None) -> None | str:
        """Export the configuration to a yaml file.

        Args:
            output_path (str, optional): The path to save the configuration file.

        Returns:
            None | str: The config file as string if output_path is None,
                otherwise None.
        """
        return self.config.export(output_path)

    def add_step(
        self,
        step_name: str,
        step_class: str | type,
        parameters: dict,
        index: int | None = None,
        substitute: bool = False,
    ) -> None:
        """Add a step to the pipeline.

        Args:
            step_name (str): Name of the step.
            step_class (str | type): Class of the step.
            parameters (dict): Parameters for the step.
            index (int, optional): Index of the step. It is only used when you
                want to add a step in a specific position. Defaults to None.
            substitute (bool, optional): Whether to substitute the step in the
                specified index. Defaults to False.
        """
        if not parameters.get("name"):
            parameters["name"] = step_name
        parameters["step_name"] = self._get_step_name(step_class)
        parameters["test"] = parameters.get("test", False)
        try:
            if index:
                if substitute:
                    self.config.add_step(index - 1, parameters, validate=True, step_class=step_class)
                    self.steps[index] = (step_name, step_class, parameters)
                else:
                    self.config.add_step(index - 1, parameters, insert=True, validate=True, step_class=step_class)
                    self.steps.insert(index, (step_name, step_class, parameters))

            else:
                self.config.add_step(None, parameters, validate=True, step_class=step_class)
                self.steps.append((step_name, step_class, parameters))
        except (ValidationError, ValueError) as e:
            msg = f"Invalid step parameters: {e}"
            self.logger.error(msg)
            raise ValueError(msg) from e

    def _get_step_name(self, step_class: type) -> str:
        """Get the name of the step class.

        Args:
            step_class (type): The class of the step.

        Returns:
            str: The name of the step class.
        """
        key_list = list(STEPS_CLASS_TYPES.keys())
        val_list = list(STEPS_CLASS_TYPES.values())
        return key_list[val_list.index(step_class)]

    def _convert_config_to_steps(self) -> list[tuple]:
        """Convert the configuration to steps.

        Returns:
            List[tuple]: The steps of the pipeline.
        """
        steps = []
        raw_step = ("raw", OpenLayer, self.config.general.to_dict(convert_path=False))
        steps.append(raw_step)
        for _, step in enumerate(self.config.steps):
            new_step = (step.name, STEPS_CLASS_TYPES[step.step_name], step.to_dict())
            steps.append(new_step)
        return steps

    def _repr_html_(self) -> str:
        """Generate HTML representation of the pipeline.

        Returns:
            str: The HTML representation of the pipeline.
        """
        return formating_html.pipeline_repr(self)
