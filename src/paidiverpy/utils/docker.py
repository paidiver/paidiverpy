"""This module contains utility functions for Docker."""

import os


def is_running_in_docker() -> bool:
    """Check if the code is running in a Docker container.

    Returns:
        bool: Whether the code is running in a Docker container.
    """
    return os.getenv("IS_DOCKER", None)
