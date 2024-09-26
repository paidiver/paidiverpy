""" Configuration module.
"""

from pathlib import Path
import json

import yaml
from paidiverpy.config.color_params import COLOR_LAYER_METHODS
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.position_params import POSITION_LAYER_METHODS
from paidiverpy.config.resample_params import RESAMPLE_LAYER_METHODS
from paidiverpy import data
from utils import DynamicConfig


class GeneralConfig(DynamicConfig):
    """General configuration class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "raw")
        self.step_name = kwargs.get("step_name", "open")
        sample_data = kwargs.get("sample_data", None)
        if sample_data:
            self._define_sample_data(sample_data)
        else:
            input_path = kwargs.get("input_path", None)
            if input_path:
                self.input_path = Path(input_path)
            self.metadata_path = kwargs.get("metadata_path", None)
            if self.metadata_path == "SAMPLE_DATA_BENTHIC":
                self.metadata_path = Path(data.load("benthic_metadata"))
            self.metadata_type = kwargs.get("metadata_type", None)
            self.image_type = kwargs.get("image_type", None)
            self.append_data_to_metadata = kwargs.get("append_data_to_metadata", False)
        output_path = kwargs.get("output_path", None)
        if output_path:
            output_path = Path(output_path)
            self.output_path = output_path
        self.n_jobs = kwargs.get("n_jobs", None)
        self.rename = kwargs.get("rename", None)
        samplings = kwargs.get("sampling", None)
        if samplings:
            self.sampling = [SamplingConfig(**sampling) for sampling in samplings]
        else:
            self.sampling = None
        converts = kwargs.get("convert", None)
        if converts:
            self.convert = [ConvertConfig(**convert) for convert in converts]
        else:
            self.convert = None
    
    def _define_sample_data(self, sample_data: str) -> None:
        """ Define the sample data.

        Args:
            sample_data (str): The sample data type
        """
        information = data.load(sample_data)
        self.input_path = Path(information["input_path"])
        self.metadata_path = Path(information["metadata_path"])
        self.metadata_type = information["metadata_type"]
        self.image_type = information["image_type"]
        if information.get("append_data_to_metadata"):
            self.append_data_to_metadata = information["append_data_to_metadata"]

class PositionConfig(DynamicConfig):
    """Position configuration class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "position")
        self.step_name = kwargs.get("step_name", "position")
        self.mode = kwargs.get("mode", None)
        if not self.mode:
            raise ValueError("The mode is not defined in the configuration file.")
        self.test = kwargs.get("test", False)
        params = kwargs.get("params", None)
        if params:
            self.params = POSITION_LAYER_METHODS[self.mode]["params"](**params)


class ConvertConfig(DynamicConfig):
    """Convert configuration class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "convert")
        self.step_name = kwargs.get("step_name", "convert")
        self.mode = kwargs.get("mode", None)
        if not self.mode:
            raise ValueError("The mode is not defined in the configuration file.")
        self.test = kwargs.get("test", False)
        params = kwargs.get("params", None)
        if params:
            self.params = CONVERT_LAYER_METHODS[self.mode]["params"](**params)


class ColorConfig(DynamicConfig):
    """Color configuration class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "color")
        self.step_name = kwargs.get("step_name", "color")
        self.mode = kwargs.get("mode", None)
        if not self.mode:
            raise ValueError("The mode is not defined in the configuration file.")
        self.test = kwargs.get("test", False)
        params = kwargs.get("params", None)

        if params:
            self.params = COLOR_LAYER_METHODS[self.mode]["params"](**params)


class SamplingConfig(DynamicConfig):
    """Sampling configuration class."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "sampling")
        self.step_name = kwargs.get("step_name", "sampling")
        self.mode = kwargs.get("mode", None)
        if not self.mode:
            raise ValueError("The mode is not defined in the configuration file.")
        self.test = kwargs.get("test", False)
        params = kwargs.get("params", None)
        if params:
            self.params = RESAMPLE_LAYER_METHODS[self.mode]["params"](**params)


config_class_mapping = {
    "general": GeneralConfig,
    "position": PositionConfig,
    "sampling": SamplingConfig,
    "color": ColorConfig,
    "convert": ConvertConfig,
}


class Configuration:
    """Configuration class.

    Args:
        config_file_path (str, optional): The configuration file path. Defaults to None.
        input_path (str, optional): The input path. Defaults to None.
        output_path (str, optional): The output path. Defaults to None.
    """

    def __init__(
        self,
        config_file_path: str = None,
        input_path: str = None,
        output_path: str = None,
    ):
        self.general = None
        self.steps = []

        if config_file_path:
            self._load_config_from_file(config_file_path)
        else:
            self._validate_paths(input_path, output_path)
            self.general = GeneralConfig(input_path=input_path, output_path=output_path)

    def _load_config_from_file(self, config_file_path: str):
        """Load the configuration from a file.

        Args:
            config_file_path (str): The configuration file path.

        Raises:
            FileNotFoundError: file not found.
            yaml.YAMLError: yaml error.
        """
        try:
            with open(config_file_path, "r", encoding="utf-8") as config_file:
                config_data = yaml.safe_load(config_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Failed to load the configuration file: {str(e)}"
            ) from e
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to load the configuration file: {str(e)}"
            ) from e

        self.general = self._validate_general_config(config_data)
        self._load_steps(config_data)

    def _validate_general_config(self, config_data: dict) -> GeneralConfig:
        """Validate the general configuration.

        Args:
            config_data (dict): The configuration data.

        Raises:
            ValueError: General configuration is not specified.
            ValueError: General configuration is empty.
            ValueError: Input and output paths are not specified.

        Returns:
            GeneralConfig: The general configuration.
        """
        if "general" not in config_data:
            raise ValueError("General configuration is not specified.")
        if not config_data["general"]:
            raise ValueError("General configuration is empty.")
        if not config_data["general"].get("input_path"):
            if not config_data["general"].get("sample_data"):
                raise ValueError("Input path is not specified.")
        if not config_data["general"].get("output_path"):
            raise ValueError("Output path is not specified.")
        name = config_data["general"].get("name")
        if not name:
            config_data["general"]["name"] = "raw"
        return GeneralConfig(**config_data["general"])

    def _validate_paths(self, input_path: str, output_path: str) -> None:
        """Validate the input and output paths.

        Args:
            input_path (str): Input path.
            output_path (str): Output path.

        Raises:
            ValueError: Input and output paths are not specified.
        """

        if not input_path or not output_path:
            raise ValueError("Input and output paths are not specified.")

    def _load_steps(self, config_data: dict) -> None:
        """Load the steps from the configuration data.

        Args:
            config_data (dict): The configuration data.

        Raises:
            ValueError: Invalid step name.
        """
        if "steps" in config_data and config_data["steps"]:
            for step_order, step in enumerate(config_data["steps"]):
                for step_name, step_config in step.items():
                    if step_name in config_class_mapping:
                        name = step_config.get("name")
                        if not name:
                            step_config["name"] = f"{step_name}_{step_order + 1}"
                        step_config["step_name"] = step_name
                        step_class = config_class_mapping[step_name]
                        step_instance = step_class(**step_config)
                        self.steps.append(step_instance)
                    else:
                        raise ValueError(f"Invalid step name: {step_name}")

    def add_config(self, config_name: str, config: dict) -> None:
        """Add a configuration.

        Args:
            config_name (str): The configuration name.
            config (dict): The configuration.

        Raises:
            ValueError: Invalid configuration name.
        """
        if config_name not in config_class_mapping:
            raise ValueError(f"Invalid configuration name: {config_name}")

        current_config = getattr(self, config_name)
        if current_config is None:
            setattr(self, config_name, config_class_mapping[config_name](**config))
        else:
            current_config.update(**config)

    def add_step(self, config_index: int = None, parameters: dict = None) -> int:
        """Add a step to the configuration.

        Args:
            config_index (int, optional): The configuration index. Defaults to None.
            parameters (dict, optional): The parameters for the step. Defaults to None.

        Raises:
            ValueError: Invalid step index.

        Returns:
            int: The step index.
        """
        if len(self.steps) == 0:
            self.steps.append(
                config_class_mapping[parameters["step_name"]](**parameters)
            )
            return len(self.steps) - 1
        if config_index is None:
            self.steps.append(
                config_class_mapping[parameters["step_name"]](**parameters)
            )
            return len(self.steps) - 1
        elif config_index < len(self.steps):
            self.steps[config_index].update(**parameters)
            return config_index
        else:
            raise ValueError(f"Invalid step index: {config_index}")

    def export(self, output_path: str) -> None:
        """Export the configuration to a file.

        Args:
            output_path (str): The output path.
        """
        with open(output_path, "w", encoding="utf-8") as config_file:
            yaml.dump(
                self.to_dict(yaml_convert=True),
                config_file,
                default_flow_style=False,
                allow_unicode=True,
            )

    def to_dict(self, yaml_convert: bool = False):
        """Convert the configuration to a dictionary.

        Args:
            yaml_convert (bool, optional): Whether to convert the configuration to a yaml format. Defaults to False.

        Returns:
            dict: The configuration as a dictionary.
        """
        result = {}
        if self.general:
            result["general"] = self.general.to_dict()
            # print(result["general"])
            # sampling = result["general"].get("sampling")
            # convert = result["general"].get("convert")
            # if sampling:
            #     print(sampling)

            #     result["general"]["sampling"] = [step.to_dict() for step in sampling]
            # if convert:
            #     result["general"]["convert"] = [step.to_dict() for step in convert]
        if yaml_convert:
            result["steps"] = [
                {step_info.pop("step_name"): step_info}
                for step in self.steps
                for step_info in [step.to_dict()]
            ]
        else:
            result["steps"] = [step.to_dict() for step in self.steps]
        return result

    def __repr__(self) -> str:
        """Return the string representation of the configuration.

        Returns:
            str: The string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=4)

    # def to_html(self):
    #     dot = graphviz.Digraph(comment='Configuration')
    #     dot.node('general', 'General')

    #     for i, step in enumerate(self.steps):
    #         step_name = step.name if hasattr(step, 'name') else f'Step {i+1}'
    #         dot.node(f'step_{i}', step_name)
    #         dot.edge('general', f'step_{i}')

    #     # Layout adjustments (optional)
    #     dot.attr(rankdir='LR')  # Left to right layout

    #     return HTML(dot.pipe('svg'))

    # def to_html(self):
    #     steps_html = ""
    #     for i, step in enumerate(self.steps):
    #         if i % 4 == 0 and i > 0:
    #             steps_html += '<div style="clear:both;"></div>'
    #         steps_html += f"""
    #             <div style="float:left; width: 100px; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;">
    #                 <h2 style="font-size:20px;">{step.name.capitalize()}</h2>
    #                 <h2 style="font-size:13px;">Type: {step.step_name.capitalize()}</h2>
    #             </div>
    #         """
    #         if i < len(self.steps) - 1:
    #             steps_html += """
    #                 <div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">
    #                     &#10132;
    #                 </div>
    #             """

    #     html = f"""
    #     <div style="display: flex; flex-wrap: wrap; align-items: center;">
    #         <div style="float:left; width: 100px; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;">
    #             <h2 style="font-size:20px;">{getattr(self.general, 'name').capitalize()}</h2>
    #             <h2 style="font-size:13px;">Type: {getattr(self.general, 'step_name').capitalize()}</h2>
    #         </div>
    #         {f'<div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">&#10132;</div>{steps_html}' if self.steps else ''}
    #     </div>
    #     """
    #     return html

    # def _repr_html_(self):
    #     return self.to_html()
