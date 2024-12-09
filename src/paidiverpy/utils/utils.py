"""Module for utility functions."""

import logging
import os
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import subprocess
from typing import List, Union

NUM_CHANNELS_RGB = 3
NUM_CHANNELS_RGBA = 4
NUM_IMAGE_DIMS = 2
DEFAULT_BITS = 8
EIGHT_BITS = 8
SIXTEEN_BITS = 16
THIRTY_TWO_BITS = 32

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""

    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format the log message with color.

        Args:
            record (logging.LogRecord): The log record.

        Returns:
            str: The formatted log message.
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def initialise_logging(verbose: int = 2) -> logging.Logger:
    """Initialise logging configuration.

    Args:
        verbose (int): Verbose level (0 = none, 1 = errors/warnings, 2 = info,
            3 = debug). Defaults to 2.

    Returns:
        logging.Logger: The logger object.
    """
    if verbose == 0:
        logging_level = logging.CRITICAL
    elif verbose == 1:
        logging_level = logging.WARNING
    elif verbose == 2:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG


    # Prepare the logging configuration arguments
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter(
        "☁ paidiverpy ☁  | %(levelname)10s | %(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logging.basicConfig(
        handlers=[handler],
        level=logging_level,
    )

    return logging.getLogger(__name__)


def raise_value_error(message: str) -> None:
    """Raise a ValueError with the given message.

    Args:
        message (str): The message to raise the ValueError with.
    """
    raise ValueError(message)

def is_running_in_docker() -> bool:
    """Check if the code is running in a Docker container.

    Returns:
        bool: Whether the code is running in a Docker container.
    """
    return os.getenv("IS_DOCKER", None)

class DynamicConfig:
    """Dynamic configuration class."""

    def update(self, **kwargs: dict) -> None:
        """Update the configuration."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self, convert_path: bool = True) -> dict:
        """Convert the configuration to a dictionary.

        Args:
            convert_path (bool, optional): Whether to convert the path to
        a string. Defaults to True.

        Returns:
            dict: The configuration as a dictionary.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                if convert_path:
                    result[key] = str(value)
                else:
                    result[key] = value
            elif isinstance(value, DynamicConfig) or issubclass(
                type(value), DynamicConfig,
            ):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, DynamicConfig) else v for v in value
                ]
            else:
                result[key] = value
        return result

def check_and_install_dependencies(dependencies: Union[List[str], None],
                                   dependencies_path: Union[str, None]) -> None:
    """Check and install dependencies.

    Args:
        dependencies (Union[List[str], None]): The dependencies to check and install.
        dependencies_path (str, None): The path to the dependencies file.

    Raises:
        PackageNotFoundError: If the package is not found.

    """
    list_of_dependencies = []
    if dependencies:
        list_of_dependencies = dependencies
    if dependencies_path:
        is_docker = is_running_in_docker()
        if is_docker:
            dependencies_filename = dependencies_path.split("/")[-1]
            dependencies_path = "/app/custom_algorithms/" + dependencies_filename
        with open(dependencies_path, "r") as file:
            list_of_dependencies += file.readlines()
    for package in list_of_dependencies:
        try:
            package_name = package.split("==")[0]
            version(package_name)
        except PackageNotFoundError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
