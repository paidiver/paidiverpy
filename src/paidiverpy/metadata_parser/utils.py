"""Utility functions for metadata parsing."""

import json
from pathlib import Path
from jsonschema import Draft202012Validator
from jsonschema import ValidationError
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.object_store import get_file_from_bucket


def validate_ifdo(file_path: str | None = None, ifdo_data: dict | None = None) -> list:
    """validate_ifdo method.

    Validates input data against iFDO scheme. Raises an exception if the
    data is invalid.

    Args:
        file_path (str): Path to the iFDO file. If not provided, ifdo_data must be.
        ifdo_data (Dict): parsed iFDO data from the file. If not provided, file_path must be.

    Returns:
        list: List of validation errors.
    """
    if not file_path and not ifdo_data:
        msg = "Either file_path or ifdo_data must be provided."
        raise ValueError(msg)
    if file_path:
        with Path(file_path).open() as file:
            ifdo_data = json.load(file)
    ifdo_version = ifdo_data.get("image-set-header", {}).get("image-set-ifdo-version", None)
    if not ifdo_version:
        msg = "No iFDO version found in metadata."
        raise ValidationError(msg)
    schema_file_path = f"https://www.marine-imaging.com/fair/schemas/ifdo-{ifdo_version}.json"
    schema = json.loads(get_file_from_bucket(schema_file_path))
    validator = Draft202012Validator(schema)
    return sorted(validator.iter_errors(ifdo_data), key=lambda e: e.path)


def convert_to_ifdo(dataset_metadata: dict, metadata: dict, output_path: str) -> None:
    """Convert metadata to iFDO format.

    Args:
        dataset_metadata (dict): Dataset metadata.
        metadata (dict): Metadata to convert.
        output_path (str): Path to save the converted metadata.
    """
    ifdo_data = {
        "image-set-header": dataset_metadata,
        "image-set-items": metadata,
    }
    errors = validate_ifdo(ifdo_data=ifdo_data)
    if errors:
        error_messages = [format_error(error.message) for error in errors]
        raise_value_error(f"Validation errors: {error_messages}")
    with Path(output_path).open("w") as file:
        json.dump(ifdo_data, file, indent=4)


def convert_to_croissant(dataset_metadata: dict, metadata: dict, output_path: str) -> None:
    """Convert metadata to Croissant format.

    Args:
        dataset_metadata (dict): Dataset metadata.
        metadata (dict): Metadata to convert.
        output_path (str): Path to save the converted metadata.
    """
    croissant_data = {
        "@context": "https://w3id.org/croissant/schema/2023-12-21/context.json",
        "name": dataset_metadata.get("title", "Unnamed Dataset"),
        "description": dataset_metadata.get("description", ""),
        "creator": {"name": dataset_metadata.get("creator", {}).get("name", ""), "email": dataset_metadata.get("creator", {}).get("email", "")},
        "license": dataset_metadata.get("license", ""),
        "keywords": dataset_metadata.get("keywords", []),
        "dataResources": [{"name": "Images", "url": metadata.get("data_url", ""), "encodingFormat": "image/jpeg"}],
    }
    with Path(output_path).open("w") as file:
        json.dump(croissant_data, file, indent=4)


def format_error(text: list) -> str:
    """Format error message.

    Args:
        text (list): List of error messages.

    Returns:
        str: Formatted error message.
    """
    if len(text) > 3:  # noqa: PLR2004
        return f"...{'.'.join(map(str, text[-3:]))}"
    return ".".join(map(str, text))
