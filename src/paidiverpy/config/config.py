"""Configuration module."""

import copy
import json
import logging
from importlib.resources import files
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from paidiverpy.config.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.custom_params import CustomParams
from paidiverpy.config.position_params import POSITION_LAYER_METHODS
from paidiverpy.config.resample_params import RESAMPLE_LAYER_METHODS
from paidiverpy.utils.data import PaidiverpyData
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.dynamic_classes import DynamicConfig
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.install_packages import check_and_install_dependencies

steps_params_mapping = {
    "colour": COLOUR_LAYER_METHODS,
    "convert": CONVERT_LAYER_METHODS,
    "position": POSITION_LAYER_METHODS,
    "sampling": RESAMPLE_LAYER_METHODS,
}

config_class_mapping = ["general", "sampling", "convert", "position", "colour"]

logger = logging.getLogger(__name__)


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
        self.metadata_conventions = kwargs.get("metadata_conventions")
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
            self.sampling = []
            for sampling in samplings:
                sampling["step_name"] = "sampling"
                sampling["name"] = "sampling"
                self.sampling.append(StepConfig(**sampling))
        else:
            self.sampling = None
        converts = kwargs.get("convert")
        if converts:
            self.convert = []
            for convert in converts:
                convert["step_name"] = "convert"
                convert["name"] = "convert"
                self.convert.append(StepConfig(**convert))
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


class StepConfig(DynamicConfig):
    """Step configuration class.

    This class is used to define the step configuration from the configuration file

    Args:
        name (str): The name of the step.
        step_name (str): The name of the step.
        **kwargs (dict): The step configuration.
    """

    def __init__(self, **kwargs: dict):
        self.name = kwargs.get("name")
        self.step_name = kwargs.get("step_name")
        self.test = kwargs.get("test", False)
        params = kwargs.get("params", {})
        if self.step_name == "custom":
            self.file_path = kwargs.get("file_path")
            self.class_name = kwargs.get("class_name")
            self.params = CustomParams(**params)
        else:
            self.mode = kwargs.get("mode")
            step_class = steps_params_mapping[self.step_name]
            self.params = step_class[self.mode]["params"](**params)


class Configuration:
    """Configuration class.

    Args:
        config_file_path (str, optional): The configuration file path. Defaults to None.
        add_general (dict, optional): The general configuration. Defaults to None.
        add_steps (dict, optional): The steps configuration. Defaults to None.
    """

    def __init__(
        self,
        config_file_path: str | None = None,
        add_general: dict | None = None,
        add_steps: list[dict] | None = None,
    ):
        self.general = None
        self.steps = []

        if config_file_path:
            self._load_config_from_file(config_file_path)
        if add_general:
            self.add_general(add_general, validate=True)
        if add_steps:
            if self.general is None:
                msg = "General configuration is not defined. Please define it first."
                logger.warning(msg)
            else:
                for step in add_steps:
                    self.add_step(None, step, validate=True)
        if not self.general and not config_file_path:
            msg = "Configuration file path or configuration parameters are not specified."
            msg += " You have to pass them manually using 'add_general' and 'add_step' functions"
            logger.warning(msg)

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
        except ValidationError as e:
            msg = f"{e!s}"
            raise ValidationError(msg) from e
        except (yaml.YAMLError, yaml.parser.ParserError) as e:
            msg = f"Failed to load the configuration file: {e!s}"
            raise yaml.YAMLError(msg) from e

        config_data["general"]["name"] = config_data["general"].get("name") or "raw"
        self.general = GeneralConfig(**config_data["general"])
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
            raise ValidationError(msg)

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
                    if step_name == "custom":
                        check_and_install_dependencies(step_config.get("dependencies"), step_config.get("dependencies_path"))
                    name = step_config.get("name")
                    if not name:
                        step_config["name"] = f"{step_name}_{step_order + 1}"
                    step_config["step_name"] = step_name
                    step_instance = StepConfig(**step_config)
                    self.steps.append(step_instance)

    def add_general(self, config: dict, validate: bool = False) -> None:
        """Add a configuration.

        Args:
            config (dict): The configuration.
            validate (bool, optional): Whether to validate the configuration. Defaults to False.

        Raises:
            ValueError: Invalid configuration name.
        """
        general_copy = copy.copy(self.general)
        if self.general is None:
            self.general = GeneralConfig(**config)
        else:
            self.general.update(**config)
        config_data = self.to_dict(yaml_convert=True)
        if validate:
            try:
                self._validate_config(config_data)
            except ValidationError as e:
                self.general = general_copy
                msg = f"Failed to validate the general config you just added: {e!s}"
                msg += "\nPlease check the parameters and try again."
                logger.error(msg)
                raise

    def add_step(self, config_index: int | None = None, parameters: dict | None = None, insert: bool = False, validate: bool = False) -> int:
        """Add a step to the configuration.

        Args:
            config_index (int, optional): The configuration index. Defaults to None.
            parameters (dict, optional): The parameters for the step. Defaults to None.
            insert (bool, optional): Whether to insert the step at the given index. Defaults to False.
            validate (bool, optional): Whether to validate the configuration. Defaults to True.

        Raises:
            ValueError: Invalid step index.

        Returns:
            int: The step index.
        """
        if self.general is None:
            msg = "General configuration is not defined. Please define it first."
            logger.warning(msg)
            raise_value_error(msg)
        copy_steps = copy.copy(self.steps)
        if len(self.steps) == 0 or config_index is None:
            self.steps.append(StepConfig(**parameters))
            config_index = len(self.steps) - 1
        elif insert and config_index < len(self.steps):
            self.steps.insert(config_index, StepConfig(**parameters))
        elif config_index < len(self.steps):
            if not parameters.get("params"):
                parameters["params"] = {}
            self.steps[config_index].update(**parameters)
        else:
            msg = f"Invalid step index: {config_index}"
            raise_value_error(msg)
        config_data = self.to_dict(yaml_convert=True)
        if validate:
            try:
                self._validate_config(config_data)
            except ValidationError as e:
                msg = f"Failed to validate the step you just added: {e!s}"
                msg += "\nPlease check the step parameters and try again."
                logger.error(msg)
                self.steps = copy_steps
                raise
        return config_index

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

    def get_output_path(self, output_path: str | None = None) -> tuple[Path | str, bool]:
        """Get the output path.

        Args:
            output_path (str, optional): The output path. Defaults to None.

        Returns:
            tuple[Path | str, bool]: The output path and whether it is remote.
        """
        if not output_path:
            output_path = self.general.output_path
        is_remote = str(output_path).startswith("s3://")
        if not is_remote:
            is_docker = is_running_in_docker()
            if is_docker:
                output_path = Path("/app/output/")
            if isinstance(output_path, str):
                output_path = Path(output_path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
        return output_path, is_remote

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
            result["steps"] = [
                {step_info.pop("step_name"): step_info} for step in self.steps for step_info in [step.to_dict()] if step_info is not None
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
