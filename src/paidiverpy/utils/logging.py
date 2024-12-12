"""Logging utilities."""

import logging
import sys


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""

    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

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
