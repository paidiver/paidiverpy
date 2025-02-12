"""Pipeline builder class for image preprocessing."""

import gc
import json
import logging
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.pipeline_params import STEPS_CLASS_TYPES
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.open_layer import OpenLayer

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
                    self.config.add_config("general", step[2])
                else:
                    self.config.add_step(None, step[2])
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
                    "Step %s does not exist. Run the pipeline fromthe beginning",
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
        else:
            self.logger.error("Invalid step format: %s", step)
            msg = f"Invalid step format: {step}"
            raise ValueError(msg)
        if isinstance(step_class, str):
            step_class = globals()[step_class]
        return step_name, step_class, step_params

    def export_config(self, output_path: str) -> None:
        """Export the configuration to a yaml file.

        Args:
            output_path (str): The path to the output file.
        """
        self.config.export(output_path)

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

        if index:
            if substitute:
                self.steps[index] = (step_name, step_class, parameters)
                self.config.add_step(index - 1, parameters)
            else:
                self.steps.insert(index, (step_name, step_class, parameters))
                self.config.add_step(index - 1, parameters)
        else:
            self.steps.append((step_name, step_class, parameters))
            self.config.add_step(None, parameters)

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

    def to_html(self) -> str:
        """Generate HTML representation of the pipeline.

        Returns:
            str: The HTML representation of the pipeline.
        """
        steps_html = ""
        parameters_html = ""

        for i, step in enumerate(self.config.steps):
            if i % 4 == 0 and i > 0:
                steps_html += '<div style="clear:both;"></div>'
            steps_html += f"""
                <div id="step_{i}" title="Click to see more information"
                    class="square" style="cursor: pointer; float:left;
                    padding: 10px; width: max-content; height: 80px;
                    margin: 10px; border: 1px solid #000; text-align: center;
                    line-height: 80px;" onclick="showParameters('step_{i}')">
                    <h2 style="font-size:20px;">{step.name.capitalize()}</h2>
                    <h2 style="font-size:13px;">Type: {step.step_name.capitalize()}</h2>
                </div>
            """
            if i < len(self.config.steps) - 1:
                steps_html += """
                    <div style="float:left; width: 50px; height: 80px;
                        margin: 10px; text-align: center; line-height: 80px;">
                        &#10132;
                    </div>
                """
            parameters_html += f"""
                <div id="parameters_step_{i}" class="parameters"
                    style="display: none;">
                    <pre>{json.dumps(step.to_dict(), indent=4)}</pre>
                </div>
            """

        general_html = f"""
        <div id="general" title="Click to see more information" class="square"
            style="float:left; cursor: pointer; padding: 10px;
            width: max-content; height: 80px; margin: 10px;
            border: 1px solid #000; text-align: center; line-height: 80px;"
            onclick="showParameters('general')">
            <h2 style="font-size:20px;">{self.config.general.name.capitalize()}</h2>
            <h2 style="font-size:13px;">Type: {self.config.general.step_name.capitalize()}</h2>
        </div>
        """

        parameters_html += f"""
            <div id="parameters_general" class="parameters" style="display: none;">
                <pre>{json.dumps(self.config.general.to_dict(), indent=4)}</pre>
            </div>
        """
        part_text = ""
        if len(self.steps) > 1:
            part_text = (
                f'<div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">&#10132;</div>{steps_html}'
            )

        return f"""
        <div style="display: flex; flex-wrap: wrap; align-items: center;">
            {general_html}{part_text}
        </div>
        <div id="parameters" style="padding: 10px; margin: 10px;">{parameters_html}</div>
        <script>
            function showParameters(id) {{
                // Hide all parameter sections
                var currentTarget = document.getElementById(id);
                var square = document.getElementsByClassName('square');
                var allParams = document.getElementsByClassName('parameters');
                var selectedParams = document.getElementById('parameters_' + id);
                var idWasVisible = false;
                if (selectedParams) {{
                    var idWasVisible = selectedParams.style.display === 'block';
                }}
                for (var i = 0; i < square.length; i++) {{
                    square[i].style.color = 'black';
                }}
                for (var i = 0; i < allParams.length; i++) {{
                    allParams[i].style.display = 'none';
                }}
                // Show the selected parameter section
                if (selectedParams) {{
                    if (idWasVisible) {{
                        selectedParams.style.display = 'none';
                        currentTarget.style.color = 'black';
                    }} else {{
                        selectedParams.style.display = 'block';
                        currentTarget.style.color = 'red';
                    }}
                }}
            }}
        </script>
        """

    def _repr_html_(self) -> str:
        """Generate HTML representation of the pipeline.

        Returns:
            str: The HTML representation of the pipeline.
        """
        return self.to_html()
