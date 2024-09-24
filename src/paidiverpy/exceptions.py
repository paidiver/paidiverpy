"""Exception classes."""

import logging


class VariableNotFound(Exception):
    """ Exception raised for when a variable is not found in the dataset.

    Args:
        Exception (Exception): The base exception class.
    """

    def __init__(self, variable_name: str) -> None:
        """Initialise the exception.
        
        Args:
            variable_name (str): The name of the variable that was not found.
        """
        message = f"Variable '{variable_name}' not found in the dataset."
        logging.warning(message)
        super().__init__(message)
