"""Pipeline builder class for image preprocessing."""

import gc
import json
import logging
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.pipeline_params import STEPS_CLASS_TYPES
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.open_layer import OpenLayer

STEP_WITHOUT_PARAMS = 2
STEP_WITH_PARAMS = 3


class Pipeline(Paidiverpy):
    """Pipeline builder class for image preprocessing.

    Args:
        config_file_path (str): The path to the configuration file.
        input_path (str): The path to the input files.
        output_path (str): The path to the output files.
        metadata_path (str): The path to the metadata file.
        metadata_type (str): The type of the metadata file.
        metadata (MetadataParser): The metadata object.
        config (Configuration): The configuration object.
        logger (logging.Logger): The logger object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        step_name (str): The name of the step.
        parameters (dict): The parameters for the step.
        config_index (int): The index of the configuration.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
        track_changes (bool): Whether to track changes. Defaults to True.
        n_jobs (int): The number of jobs to run in parallel.
    """

    def __init__(
        self,
        config_file_path: str | None = None,
        steps: list[tuple] | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        metadata_path: str | None = None,
        metadata_type: str | None = None,
        metadata: MetadataParser = None,
        config: Configuration = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
        track_changes: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata_type=metadata_type,
            metadata=metadata,
            config=config,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
            track_changes=track_changes,
            n_jobs=n_jobs,
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

    def run(self, from_step: int | None = None) -> None:
        """Run the pipeline.

        Args:
            from_step (int, optional): The step to start from. Defaults to None,
        which means the pipeline will start from the last runned step.

        Raises:
            ValueError: No steps defined for the pipeline
            ValueError: Invalid step format
        """
        if not self.steps:
            self.logger.error("No steps defined for the pipeline")
            msg = "No steps defined for the pipeline"
            raise ValueError(msg)
        if from_step is not None:
            if len(self.images.images) > from_step:
                self.runned_steps = from_step
                self.clear_steps(from_step + 1)
            else:
                self.logger.warning(
                    "Step %s does not exist. Run the pipeline from the beginning",
                    from_step,
                )
        for index, step in enumerate(self.steps):
            if index > self.runned_steps:
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
                self.logger.info(
                    "Running step %s: %s - %s", index, step_name, step_class.__name__,
                )
                step_params["step_name"] = self._get_step_name(step_class)
                step_params["name"] = step_name
                if step_name == "raw":
                    step_instance = step_class(
                        step_name=step_name,
                        config=self.config,
                        metadata=self.metadata,
                        parameters=step_params,
                        track_changes=self.track_changes,
                        n_jobs=self.n_jobs,
                    )
                else:
                    step_instance = step_class(
                        config=self.config,
                        metadata=self.metadata,
                        images=self.images,
                        step_name=step_name,
                        parameters=step_params,
                        config_index=index - 1,
                        track_changes=self.track_changes,
                        n_jobs=self.n_jobs,
                    )
                step_instance.run()
                if not step_params.get("test", False):
                    self.images = step_instance.images
                    self.set_metadata(step_instance.get_metadata(flag="all"))
                    self.runned_steps = index
                self.logger.info("Step %s completed", index)

                del step_instance
                gc.collect()

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
            step_class (Union[str, type]): Class of the step.
            parameters (dict): Parameters for the step.
            index (int, optional): Index of the step. It is only used when you want
        to add a step in a specific position. Defaults to None.
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
                <div id="step_{i}" title="Click to see more information" class="square" style="cursor: pointer; float:left; padding: 10px; width: max-content; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;" onclick="showParameters('step_{i}')">
                    <h2 style="font-size:20px;">{step.name.capitalize()}</h2>
                    <h2 style="font-size:13px;">Type: {step.step_name.capitalize()}</h2>
                </div>
            """
            if i < len(self.config.steps) - 1:
                steps_html += """
                    <div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">
                        &#10132;
                    </div>
                """
            parameters_html += f"""
                <div id="parameters_step_{i}" class="parameters" style="display: none;">
                    <pre>{json.dumps(step.to_dict(), indent=4)}</pre>
                </div>
            """

        general_html = f"""
        <div id="general" title="Click to see more information" class="square" style="float:left; cursor: pointer; padding: 10px; width: max-content; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;" onclick="showParameters('general')">
            <h2 style="font-size:20px;">{self.config.general.name.capitalize()}</h2>
            <h2 style="font-size:13px;">Type: {self.config.general.step_name.capitalize()}</h2>
        </div>
        """

        parameters_html += f"""
            <div id="parameters_general" class="parameters" style="display: none;">
                <pre>{json.dumps(self.config.general.to_dict(), indent=4)}</pre>
            </div>
        """

        return f"""
        <div style="display: flex; flex-wrap: wrap; align-items: center;">
            {general_html}
            {f'<div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">&#10132;</div>{steps_html}' if len(self.steps) > 1 else ''}
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
