import gc
import json
from paidiverpy.open_layer import OpenLayer
from paidiverpy import Paidiverpy
from paidiverpy.pipeline.params import STEPS_CLASS_TYPES


class Pipeline(Paidiverpy):
    def __init__(
        self,
        config_file_path=None,
        steps=None,
        input_path=None,
        output_path=None,
        catalog_path=None,
        catalog_type=None,
        catalog=None,
        config=None,
        logger=None,
        raise_error=False,
        verbose=True,
    ):

        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            catalog_path=catalog_path,
            catalog_type=catalog_type,
            catalog=catalog,
            config=config,
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

    def run(self, from_step=None):
        # Ensure steps are provided
        if not self.steps:
            self.logger.error("No steps defined for the pipeline")
            raise ValueError("No steps defined for the pipeline")
        if from_step is not None:
            if len(self.images.images) > from_step:
                self.runned_steps = from_step
                self.clear_steps(from_step + 1)
            else:
                self.logger.warning(
                    f"Step {from_step} does not exist. Run the pipeline from the beginning"
                )
        for index, step in enumerate(self.steps):
            if index > self.runned_steps:
                if len(step) == 2:
                    # When no parameters are provided
                    step_name, step_class = step
                    step_params = {}
                elif len(step) == 3:
                    # When parameters are provided
                    step_name, step_class, step_params = step
                else:
                    self.logger.error(f"Invalid step format: {step}")
                    raise ValueError(f"Invalid step format: {step}")
                if isinstance(step_class, str):
                    # If step_class is a string, import the class
                    step_class = globals()[step_class]
                self.logger.info(
                    f"Running step {index}: {step_name} - {step_class.__name__}"
                )
                # Instantiate the class
                step_params["step_name"] = self._get_step_name(step_class)
                step_params["name"] = step_name
                if step_name == "raw":
                    step_instance = step_class(
                        step_name=step_name,
                        config=self.config,
                        catalog=self.catalog,
                        parameters=step_params,
                    )
                else:
                    step_instance = step_class(
                        config=self.config,
                        catalog=self.catalog,
                        images=self.images,
                        step_name=step_name,
                        parameters=step_params,
                        config_index=index - 1,
                    )
                step_instance.run()
                self.logger.info(f"Step {index} completed")
                if not step_params.get("test", False):
                    self.images = step_instance.images
                    self.set_catalog(step_instance.get_catalog(flag="all"))
                    self.runned_steps = index
                    self.logger.info(f"Step {index} saved")

                del step_instance
                gc.collect()

    def export_config(self, output_path: str):
        self.config.export(output_path)

    def add_step(self, step_name, step_class, parameters, index=None, substitute=False):
        parameters["name"] = (
            step_name if not parameters.get("name") else parameters["name"]
        )
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

    def _get_step_name(self, step_class):
        key_list = list(STEPS_CLASS_TYPES.keys())
        val_list = list(STEPS_CLASS_TYPES.values())
        return key_list[val_list.index(step_class)]

    def _convert_config_to_steps(self):
        steps = []
        raw_step = ("raw", OpenLayer, self.config.general.to_dict(convert_path=False))
        steps.append(raw_step)
        for _, step in enumerate(self.config.steps):
            new_step = (step.name, STEPS_CLASS_TYPES[step.step_name], step.to_dict())
            steps.append(new_step)
        return steps

    def to_html(self):
        steps_html = ""
        parameters_html = ""

        # Generate HTML for steps
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
            # Generate hidden parameter sections
            parameters_html += f"""
                <div id="parameters_step_{i}" class="parameters" style="display: none;">
                    <pre>{json.dumps(step.to_dict(), indent=4)}</pre>
                </div>
            """

        # General step HTML
        general_html = f"""
        <div id="general" title="Click to see more information" class="square" style="float:left; cursor: pointer; padding: 10px; width: max-content; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;" onclick="showParameters('general')">
            <h2 style="font-size:20px;">{getattr(self.config.general, 'name').capitalize()}</h2>
            <h2 style="font-size:13px;">Type: {getattr(self.config.general, 'step_name').capitalize()}</h2>
        </div>
        """

        # General parameters HTML
        parameters_html += f"""
            <div id="parameters_general" class="parameters" style="display: none;">
                <pre>{json.dumps(self.config.general.to_dict(), indent=4)}</pre>
            </div>
        """

        # Complete HTML
        html = f"""
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
        return html

    def _repr_html_(self):
        return self.to_html()
