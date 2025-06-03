"""Logging utilities."""

import logging
import sys
from enum import IntEnum
from typing import ClassVar
from paidiverpy.utils.exceptions import raise_value_error


class VerboseLevel(IntEnum):
    """Verbose levels for logging."""

    NONE = 0
    ERRORS_WARNINGS = 1
    INFO = 2
    DEBUG = 3


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

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


def initialise_logging(verbose: int = 2, logger_name: str = "paidiverpy") -> logging.Logger:
    """Initialise logging configuration.

    Args:
        verbose (int): Verbose level (0 = NONE, 1 = ERRORS_WARNINGS, 2 = INFO, 3 = DEBUG).
            Defaults to 2.
        logger_name (str): The name of the logger. Defaults to "paidiverpy".

    Returns:
        logging.Logger: The logger object.
    """
    try:
        log_level = {
            VerboseLevel.NONE: logging.CRITICAL,
            VerboseLevel.ERRORS_WARNINGS: logging.WARNING,
            VerboseLevel.INFO: logging.INFO,
            VerboseLevel.DEBUG: logging.DEBUG,
        }[VerboseLevel(verbose)]
    except ValueError as err:
        msg = f"Invalid verbose level: {verbose}. Choose from {list(VerboseLevel)}."
        raise ValueError(msg) from err

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter(
        "☁ paidiverpy ☁  | %(levelname)10s | %(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setLevel(log_level)  # Set level on the handler too

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(handler)

    return logger


def check_raise_error(raise_error: bool, message: str) -> None:
    """Check if an error should be raised and raise it if necessary.

    Args:
        raise_error (bool): Whether to raise an error.
        message (str): The error message.

    Raises:
        ValueError: The error message.
    """
    if raise_error:
        logging.error(message)
        raise_value_error(message)
    logging.warning(message)
