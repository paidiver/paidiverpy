"""Resample parameters dataclasses.

This module contains the dataclasses for the parameters of the convert layer
functions.
"""

from paidiverpy.utils.dynamic_classes import DynamicConfig


class CustomParams(DynamicConfig):
    """This class contains the parameters for the obscure resampling."""

    def __init__(self, **kwargs: dict) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
