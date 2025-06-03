"""Schema JSON handler module."""

import argparse
import copy
import json
from pathlib import Path
from paidiverpy.config.configuration import config_name_mapping
from paidiverpy.models.config_model import ConfigModel
from paidiverpy.utils.logging_functions import initialise_logging

logger = initialise_logging()


def generate_schema(output_path: str) -> None:
    """Generate the schema for the configuration model.

    Args:
        output_path (str): The path to save the schema.
    """
    schema = ConfigModel.model_json_schema()
    for step_type in config_name_mapping:
        target_ref = f"#/$defs/{step_type.capitalize()}Config"
        schema = wrap_ref(schema, target_ref=target_ref, wrapper_key=step_type)

    with Path(output_path).open("w") as f:
        json.dump(schema, f, indent=4)
    logger.info("Schema saved to %s", output_path)


def wrap_ref(schema: dict, target_ref: str, wrapper_key: str) -> dict:
    """Wrap a reference in the schema with a key.

    Args:
        schema (dict): The original schema.
        target_ref (str): The reference to wrap.
        wrapper_key (str): The key to wrap the reference with.

    Returns:
        dict: The modified schema with the reference wrapped.
    """

    def _transform(obj: dict | list | str | float | bool, parent_key: str = "") -> dict:
        """Recursively transform the schema.

        Args:
            obj (dict | list | str | float | bool): The object to transform.
            parent_key (str): The parent key in the schema.

        Returns:
            dict: The transformed object.
        """
        if isinstance(obj, dict):
            if obj.get("$ref") == target_ref and parent_key != "items":
                return {"type": "object", "properties": {wrapper_key: {"$ref": target_ref}}, "required": [wrapper_key], "additionalProperties": False}

            return {k: _transform(v, parent_key=k) for k, v in obj.items()}

        if isinstance(obj, list):
            return [_transform(i, parent_key=parent_key) for i in obj]

        return obj

    return _transform(copy.deepcopy(schema))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON schema for Paidiverpy configuration.")
    parser.add_argument("output_path", help="Path to save the generated JSON schema file")
    args = parser.parse_args()
    generate_schema(args.output_path)
