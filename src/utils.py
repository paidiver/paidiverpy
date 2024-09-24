""" Module for utility functions.
"""
import logging
import multiprocessing
from pathlib import Path
import sys


def initialise_logging(verbose: int = 2) -> logging.Logger:
    """Initialise logging configuration.

    Args:
        verbose (int): Verbose level (0 = none, 1 = errors/warnings, 2 = info).

    Returns:
        logging.Logger: The logger object.
    """
    if verbose == 0:
        logging_level = logging.CRITICAL
    elif verbose == 1:
        logging_level = logging.WARNING
    else:
        logging_level = logging.INFO

    logging.basicConfig(
        stream=sys.stdout,
        format=("☁ paidiverpy ☁  | %(levelname)10s | "
                "%(asctime)s | %(message)s"),
        level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_n_jobs(n_jobs: int) -> int:
    """Determine the number of jobs based on n_jobs parameter.

    Args:
        n_jobs (int): The number of n_jobs.

    Returns:
        int: The number of jobs to use.
    """

    if n_jobs == -1:
        return multiprocessing.cpu_count()
    if n_jobs > 1:
        return min(n_jobs, multiprocessing.cpu_count())
    return 1


class DynamicConfig:
    """Dynamic configuration class."""

    def update(self, **kwargs):
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
            elif isinstance(value, DynamicConfig):
                result[key] = value.to_dict()
            elif issubclass(type(value), DynamicConfig):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, DynamicConfig) else v for v in value
                ]
            else:
                result[key] = value
        return result


# class ClassInstanceMethod:
#     def __init__(self, func):
#         self.func = func

#     def __get__(self, instance, cls):
#         if instance is not None:
#             return lambda *args, **kwargs: self.func(instance, *args, **kwargs)
#         return lambda *args, **kwargs: self.func(cls, *args, **kwargs)
