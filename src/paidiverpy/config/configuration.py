"""Configuration module."""

import copy
import json
from importlib.resources import files
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.models.step_config import ColourConfig
from paidiverpy.models.step_config import ConvertConfig
from paidiverpy.models.step_config import CustomConfig
from paidiverpy.models.step_config import PositionConfig
from paidiverpy.models.step_config import SamplingConfig
from paidiverpy.models.step_config import StepConfig
from paidiverpy.utils import formating_html
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.install_packages import check_and_install_dependencies
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.object_store import get_file_from_bucket
from paidiverpy.utils.object_store import path_is_remote

logger = initialise_logging()

config_name_mapping = {
    "colour": ColourConfig,
    "convert": ConvertConfig,
    "position": PositionConfig,
    "sampling": SamplingConfig,
    "custom": CustomConfig,
}


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
        self.is_remote = False
        self.output_is_remote = False
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
            Configuration.validate_config(config_data)
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
        self._update_remote_options()
        self._load_steps(config_data)

    @staticmethod
    def validate_config(config: dict | str | Path, local: bool = True) -> None:
        """Validate the configuration.

        Args:
            config (dict | str | Path): The configuration to validate.
            local (bool, optional): Whether the schema is local. Defaults to True.
        """
        schema_json_remote = "https://raw.githubusercontent.com/paidiver/paidiverpy/refs/heads/main/src/paidiverpy/configuration-schema.json"
        schema_file_path = files("paidiverpy").joinpath("configuration-schema.json") if local else schema_json_remote
        if path_is_remote(schema_file_path):
            schema = get_file_from_bucket(schema_file_path)
            schema = json.loads(schema.decode("utf-8"))
        else:
            with schema_file_path.open("r", encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
        validator = Draft202012Validator(schema)
        if isinstance(config, str | Path):
            with Path(str(config)).open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
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
                    config_class = config_name_mapping.get(step_name)
                    step_instance = config_class(**step_config)
                    self.steps.append(step_instance)

    def _update_remote_options(self) -> None:
        """Update the remote options in the general configuration."""
        if self.general:
            self.is_remote = path_is_remote(str(self.general.input_path)) if self.general.input_path else False
            self.output_is_remote = path_is_remote(str(self.general.output_path)) if self.general.output_path else False

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
                Configuration.validate_config(config_data)
            except ValidationError as e:
                self.general = general_copy
                msg = f"Failed to validate the general config you just added: {e!s}"
                msg += "\nPlease check the parameters and try again."
                logger.error(msg)
                raise ValidationError(msg) from e

    def add_step(
        self,
        config_index: int | None = None,
        parameters: dict | None = None,
        insert: bool = False,
        validate: bool = False,
        step_class: BaseModel | None = None,
    ) -> int:
        """Add a step to the configuration.

        Args:
            config_index (int, optional): The configuration index. Defaults to None.
            parameters (dict, optional): The parameters for the step. Defaults to None.
            insert (bool, optional): Whether to insert the step at the given index. Defaults to False.
            validate (bool, optional): Whether to validate the configuration. Defaults to True.
            step_class (BaseModel, optional): The class of the step. Defaults to None.

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
        step_class = config_name_mapping.get(step_class.__name__.replace("Layer", "").lower()) if step_class else StepConfig
        if len(self.steps) == 0 or config_index is None:
            self.steps.append(step_class(**parameters))
            config_index = len(self.steps) - 1
        elif insert and config_index < len(self.steps):
            self.steps.insert(config_index, step_class(**parameters))
        elif config_index < len(self.steps):
            if not parameters.get("params"):
                parameters["params"] = {}
            self.steps[config_index] = step_class(**parameters)
        else:
            msg = f"Invalid step index: {config_index}"
            raise_value_error(msg)
        config_data = self.to_dict(yaml_convert=True)
        if validate:
            try:
                Configuration.validate_config(config_data)
            except ValidationError as e:
                msg = f"Failed to validate the step you just added: {e!s}"
                msg += "\nPlease check the step parameters and try again."
                logger.error(msg)
                self.steps = copy_steps
                raise
        return config_index

    def export(self, output_path: str | None) -> None | str:
        """Export the configuration to a file.

        Args:
            output_path (str, optional): The path to save the configuration file. If None, returns the configuration as a YAML string.

        Returns:
            None | str: If output_path is None, returns the configuration as a YAML string.
                        Otherwise, writes the configuration to the specified file.
        """
        if not output_path:
            return yaml.dump(
                self.to_dict(yaml_convert=True),
                default_flow_style=False,
                allow_unicode=True,
            )
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as config_file:
            yaml.dump(
                self.to_dict(yaml_convert=True),
                config_file,
                default_flow_style=False,
                allow_unicode=True,
            )
            return None

    def get_output_path(self, output_path: str | None = None) -> tuple[Path | str, bool]:
        """Get the output path.

        Args:
            output_path (str, optional): The output path. Defaults to None.

        Returns:
            tuple[Path | str, bool]: The output path and whether it is remote.
        """
        if not output_path:
            output_path = self.general.output_path
        is_remote = path_is_remote(str(output_path))
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

    def _repr_html_(self) -> str:
        """Generate HTML representation of the configuration.

        Returns:
            str: The HTML representation of the configuration.
        """
        return formating_html.config_repr(self)
