"""This module contains functions to check and install dependencies."""

import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from paidiverpy.utils.docker import is_running_in_docker

PACKAGE_REGEX = re.compile(r"^[a-zA-Z0-9_-]+(==[a-zA-Z0-9_.-]+)?$")


def check_and_install_dependencies(dependencies: str | None, dependencies_path: str | None) -> None:
    """Check and install dependencies.

    Args:
        dependencies (str, None): The dependencies to check and install.
        dependencies_path (str, None): The path to the dependencies file.

    Raises:
        PackageNotFoundError: If the package is not found.

    """
    list_of_dependencies = dependencies.split(",") if dependencies else []
    if dependencies_path:
        is_docker = is_running_in_docker()
        if is_docker:
            dependencies_filename = dependencies_path.split("/")[-1]
            dependencies_path = "/app/custom_algorithms/" + dependencies_filename
        dependencies_path = Path(dependencies_path)
        with Path.open(dependencies_path) as file:
            list_of_dependencies += file.readlines()
    for package in list_of_dependencies:
        package_name = package.strip()
        if not PACKAGE_REGEX.match(package_name):
            msg = f"Invalid package name or version: {package_name}"
            raise ValueError(msg)
        package_name = package_name.split("==")[0]
        if not is_package_installed(package_name):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])  # noqa: S603


def is_package_installed(package_name: str) -> bool:
    """Check if the package is installed.

    Args:
        package_name (str): The package name.

    Returns:
        bool: Whether the package is installed.
    """
    try:
        version(package_name)
    except PackageNotFoundError:
        return False
    return True
