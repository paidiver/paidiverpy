"""Dynamic classes for configuration."""

from pathlib import Path
from typing import Any


class DynamicConfig:
    """Dynamic configuration class."""

    def update(self, **kwargs: dict) -> None:
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
            if value is None or value == {}:
                continue
            if isinstance(value, dict):
                result[key] = {}
                for k, v in value.items():
                    result = self._update_dict(result, k, v, convert_path, main_key=key)
            else:
                result = self._update_dict(result, key, value, convert_path)
        return result

    def _update_dict(self, result: dict, key: str, value: Any, convert_path: bool, main_key: str | None = None) -> dict:  # noqa: ANN401
        """Update the dictionary with the configuration.

        Args:
            result (dict): The result dictionary.
            key (str): The key to update.
            value (Any): The value to update.
            convert_path (bool): Whether to convert the path to a string.
            main_key (str | None, optional): The main key to update. Defaults to None.

        Returns:
            dict: The updated dictionary.
        """
        updated_result = {}
        if isinstance(value, Path):
            if convert_path:
                updated_result[key] = str(value)
            else:
                updated_result[key] = value
        elif isinstance(value, DynamicConfig) or issubclass(
            type(value),
            DynamicConfig,
        ):
            updated_result[key] = value.to_dict()
        elif isinstance(value, list):
            updated_result[key] = [v.to_dict() if isinstance(v, DynamicConfig) else v for v in value if v is not None]
        else:
            updated_result[key] = value
        if main_key:
            result[main_key][key] = updated_result[key]
        else:
            result[key] = updated_result[key]
        return result
