"""Custom parameters dataclasses.

This module contains the dataclasses for the parameters used in the custom_params module.
"""

from paidiverpy.utils.dynamic_classes import DynamicConfig


class CustomParams(DynamicConfig):
    """This class contains the parameters for the custom_params module."""

    def __init__(self, **kwargs: dict) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    raise_error: bool = False
