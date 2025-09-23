"""Paidiverpy base package."""

__version__ = "0.2.1"
__author__ = "Tobias Ferreira"
__credits__ = "National Oceanography Centre"
from .paidiverpy import Paidiverpy
from .utils.locals import show_versions

__all__ = ("Paidiverpy", "show_versions")
