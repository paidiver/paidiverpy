"""Configuration module."""

import json
from importlib.resources import files
from pathlib import Path
import jsonschema
import yaml
from jsonschema import Draft202012Validator
from paidiverpy.config.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.custom_params import CustomParams
from paidiverpy.config.position_params import POSITION_LAYER_METHODS
from paidiverpy.config.resample_params import RESAMPLE_LAYER_METHODS
from paidiverpy.utils.data import PaidiverpyData
from paidiverpy.utils.dynamic_classes import DynamicConfig
from paidiverpy.utils.install_packages import check_and_install_dependencies


class GeneralConfig(DynamicConfig):
    """General configuration class.

    This class is used to define the general configuration from the configuration file
        or from the input from the user.

    """

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name", "raw")
        self.step_name = kwargs.get("step_name", "open")
        self.sample_data = kwargs.get("sample_data")
        if self.sample_data:
            self._define_sample_data(self.sample_data)
            self.is_remote = False
        else:
            input_path = kwargs.get("input_path")
            self.is_remote = str(input_path).startswith(("http://", "https://", "s3://"))
            if input_path:
                self.input_path = Path(input_path) if not self.is_remote else input_path
            self.metadata_path = kwargs.get("metadata_path")
            self.metadata_type = kwargs.get("metadata_type")
            self.image_type = kwargs.get("image_type")
            self.append_data_to_metadata = kwargs.get("append_data_to_metadata", False)
        output_path = kwargs.get("output_path")
        self.output_is_remote = str(output_path).startswith(("http://", "https://", "s3://"))
        if output_path:
            if not self.output_is_remote:
                output_path = Path(output_path)
            self.output_path = output_path

        self.n_jobs = kwargs.get("n_jobs", 1)
        self.client = kwargs.get("client")
        self.track_changes = kwargs.get("track_changes", True)
        self.rename = kwargs.get("rename")
        samplings = kwargs.get("sampling")
        if samplings:
            self.sampling = [SamplingConfig(**sampling) for sampling in samplings]
        else:
            self.sampling = None
        converts = kwargs.get("convert")
        if converts:
            self.convert = [ConvertConfig(**convert) for convert in converts]
        else:
            self.convert = None

    def _define_sample_data(self, sample_data: str) -> None:
        """Define the sample data.

        Args:
            sample_data (str): The sample data type
        """
        data = PaidiverpyData()
        information = data.load(sample_data)
        self.input_path = Path(information["input_path"])
        self.metadata_path = Path(information["metadata_path"])
        self.metadata_type = information["metadata_type"]
        self.image_type = information["image_type"]
        if information.get("append_data_to_metadata"):
            self.append_data_to_metadata = information["append_data_to_metadata"]


class PositionConfig(DynamicConfig):
    """Position configuration class."""

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name", "position")
        self.step_name = kwargs.get("step_name", "position")
        self.mode = kwargs.get("mode")
        if not self.mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        self.test = kwargs.get("test", False)
        params = kwargs.get("params")
        if params:
            self.params = POSITION_LAYER_METHODS[self.mode]["params"](**params)


class ConvertConfig(DynamicConfig):
    """Convert configuration class."""

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name", "convert")
        self.step_name = kwargs.get("step_name", "convert")
        self.mode = kwargs.get("mode")
        if not self.mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        self.test = kwargs.get("test", False)
        params = kwargs.get("params")
        if params:
            self.params = CONVERT_LAYER_METHODS[self.mode]["params"](**params)


class ColourConfig(DynamicConfig):
    """Colour configuration class."""

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name", "colour")
        self.step_name = kwargs.get("step_name", "colour")
        self.mode = kwargs.get("mode")
        if not self.mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        self.test = kwargs.get("test", False)
        params = kwargs.get("params")

        if params:
            self.params = COLOUR_LAYER_METHODS[self.mode]["params"](**params)


class SamplingConfig(DynamicConfig):
    """Sampling configuration class."""

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name", "sampling")
        self.step_name = kwargs.get("step_name", "sampling")
        self.mode = kwargs.get("mode")
        if not self.mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        self.test = kwargs.get("test", False)
        params = kwargs.get("params")
        if params:
            self.params = RESAMPLE_LAYER_METHODS[self.mode]["params"](**params)


class CustomConfig(DynamicConfig):
    """Sampling configuration class."""

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name", "custom")
        self.step_name = kwargs.get("step_name", "custom")
        self.file_path = kwargs.get("file_path")
        self.class_name = kwargs.get("class_name")
        if not self.file_path or not self.class_name:
            msg = "The file_path and the class_na,e is not defined in the configuration file."
            raise ValueError(msg)
        self.test = kwargs.get("test", False)
        params = kwargs.get("params")
        if params:
            self.params = CustomParams(**params)


config_class_mapping = {
    "general": GeneralConfig,
    "position": PositionConfig,
    "sampling": SamplingConfig,
    "colour": ColourConfig,
    "convert": ConvertConfig,
    "custom": CustomConfig,
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
        config_file_path: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
    ):
        self.general = None
        self.steps = []

        if config_file_path:
            self._load_config_from_file(config_file_path)
        else:
            self._validate_paths(input_path, output_path)
            self.general = GeneralConfig(input_path=input_path, output_path=output_path)

    def _load_config_from_file(self, config_file_path: str) -> None:
        """Load the configuration from a file.

        Args:
            config_file_path (str): The configuration file path.

        Raises:
            FileNotFoundError: file not found.
            yaml.YAMLError: yaml error.
        """
        try:
            config_file_path = Path(config_file_path)
            with config_file_path.open(encoding="utf-8") as config_file:
                config_data = yaml.safe_load(config_file)
            self._validate_config(config_data)
        except FileNotFoundError as e:
            msg = f"Failed to load the configuration file: {e!s}"
            raise FileNotFoundError(msg) from e
        except jsonschema.exceptions.ValidationError as e:
            msg = f"{e!s}"
            raise jsonschema.exceptions.ValidationError(msg) from e
        except yaml.YAMLError as e:
            msg = f"Failed to load the configuration file: {e!s}"
            raise yaml.YAMLError(msg) from e
        except yaml.parser.ParserError as e:
            msg = f"Failed to parse the configuration file: {e!s}"
            raise yaml.parser.ParserError(msg) from e

        self.general = self._validate_general_config(config_data)
        self._load_steps(config_data)

    def _validate_config(self, config: dict) -> None:
        """Validate the configuration.

        Args:
            config (dict): The configuration.
        """
        schema_file_path = files("paidiverpy").joinpath("configuration-schema.json")
        with schema_file_path.open("r", encoding="utf-8") as schema_file:
            schema = json.load(schema_file)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
        if errors:
            msg = "Failed to validate the configuration file.\n"
            for error in errors:
                msg += f"{error}: {error.message}\n"
            raise jsonschema.exceptions.ValidationError(msg)

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
            msg = "General configuration is not specified."
            raise ValueError(msg)
        if not config_data["general"]:
            msg = "General configuration is empty."
            raise ValueError(msg)
        if not config_data["general"].get("input_path") and not config_data["general"].get("sample_data"):
            msg = "Input path is not specified."
            raise ValueError(msg)
        if not config_data["general"].get("output_path"):
            msg = "Output path is not specified."
            raise ValueError(msg)
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
            msg = "Input and output paths are not specified."
            raise ValueError(msg)

    def _load_steps(self, config_data: dict) -> None:
        """Load the steps from the configuration data.

        Args:
            config_data (dict): The configuration data.

        Raises:
            ValueError: Invalid step name.
        """
        if config_data.get("steps"):
            for step_order, step in enumerate(config_data["steps"]):
                for step_name, step_config in step.items():
                    if step_name in config_class_mapping:
                        if step_name == "custom":
                            check_and_install_dependencies(step_config.get("dependencies"), step_config.get("dependencies_path"))
                        name = step_config.get("name")
                        if not name:
                            step_config["name"] = f"{step_name}_{step_order + 1}"
                        step_config["step_name"] = step_name
                        step_class = config_class_mapping[step_name]
                        step_instance = step_class(**step_config)
                        self.steps.append(step_instance)
                    else:
                        msg = f"Invalid step name: {step_name}"
                        raise ValueError(msg)

    def add_config(self, config_name: str, config: dict) -> None:
        """Add a configuration.

        Args:
            config_name (str): The configuration name.
            config (dict): The configuration.

        Raises:
            ValueError: Invalid configuration name.
        """
        if config_name not in config_class_mapping:
            msg = f"Invalid configuration name: {config_name}"
            raise ValueError(msg)

        current_config = getattr(self, config_name)
        if current_config is None:
            setattr(self, config_name, config_class_mapping[config_name](**config))
        else:
            current_config.update(**config)

    def add_step(self, config_index: int | None = None, parameters: dict | None = None) -> int:
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
            self.steps.append(config_class_mapping[parameters["step_name"]](**parameters))
            return len(self.steps) - 1
        if config_index is None:
            self.steps.append(config_class_mapping[parameters["step_name"]](**parameters))
            return len(self.steps) - 1
        if config_index < len(self.steps):
            self.steps[config_index].update(**parameters)
            return config_index
        msg = f"Invalid step index: {config_index}"
        raise ValueError(msg)

    def export(self, output_path: str) -> None:
        """Export the configuration to a file.

        Args:
            output_path (str): The output path.
        """
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as config_file:
            yaml.dump(
                self.to_dict(yaml_convert=True),
                config_file,
                default_flow_style=False,
                allow_unicode=True,
            )

    def to_dict(self, yaml_convert: bool = False) -> dict:
        """Convert the configuration to a dictionary.

        Args:
            yaml_convert (bool, optional): Whether to convert the configuration to a yaml format. Defaults to False.

        Returns:
            dict: The configuration as a dictionary.
        """
        result = {}
        if self.general:
            result["general"] = self.general.to_dict()
        if yaml_convert:
            result["steps"] = [{step_info.pop("step_name"): step_info} for step in self.steps for step_info in [step.to_dict()]]
        else:
            result["steps"] = [step.to_dict() for step in self.steps]
        return result

    def __repr__(self) -> str:
        """Return the string representation of the configuration.

        Returns:
            str: The string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=4)
