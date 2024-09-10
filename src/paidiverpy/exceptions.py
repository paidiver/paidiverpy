"""Exception classes."""

import logging


class VariableNotFound(Exception):

    def __init__(self, variable_name):
        """Initialise the exception."""
        message = f"Variable '{variable_name}' not found in the dataset."
        logging.warning(message)
        super().__init__(message)


class DimensionMismatch(Exception):

    def __init__(self, dim, size, expected_size):
        """Initialise the exception."""
        message = f"Dimension {dim} has size {size}, expected {expected_size}."
        logging.warning(message)
        super().__init__(message)


class ExpectedAttrsNotFound(Exception):
    """Exception raised for when expected attributes are not found in the metadata."""

    def __init__(self, expected_attrs):
        """Initialise the exception."""
        message = f"Expected {expected_attrs} not found in metadata."
        logging.warning(message)
        super().__init__(message)
