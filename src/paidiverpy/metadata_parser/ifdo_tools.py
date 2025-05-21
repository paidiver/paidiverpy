"""Utility functions for metadata parsing."""

import ast
import json
import logging
import re
from pathlib import Path
import pandas as pd
from jsonschema import Draft202012Validator
from jsonschema import ValidationError
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
    errors = sorted(validator.iter_errors(ifdo_data), key=lambda e: e.path)
    errors = [
        {
            "path": list(error.absolute_path),
            "message": error.message,
        }
        for error in errors
    ]

    return parse_validation_errors(errors, schema)


def convert_to_ifdo(dataset_metadata: dict, metadata: dict, output_path: str) -> None:
    """Convert metadata to iFDO format.

    Args:
        dataset_metadata (dict): Dataset metadata.
        metadata (dict): Metadata to convert.
        output_path (str): Path to save the converted metadata.
    """
    ifdo_version = dataset_metadata.get("image-set-ifdo-version", "v2.1.0")
    schema_file_path = f"https://www.marine-imaging.com/fair/schemas/ifdo-{ifdo_version}.json"
    ifdo_schema = json.loads(get_file_from_bucket(schema_file_path))
    image_set_header, missing_fields_header = parse_ifdo_header(dataset_metadata, ifdo_schema, metadata)
    for col in metadata.select_dtypes(include=["datetime64[ns]"]).columns:
        metadata[col] = metadata[col].astype(str)
    image_set_header["image-set-ifdo-version"] = ifdo_version
    image_set_items, missing_fields_items = parse_ifdo_items(metadata, ifdo_schema)
    if missing_fields_header or missing_fields_items:
        logging.warning("Missing required fields in iFDO header or items")
        logging.warning("You need to set then in the metadata or dataset_metadata arguments")
        logging.warning("The missing fields will be set to the description value on the iFDO schema file")
    if missing_fields_header:
        logging.warning("Missing fields in iFDO header: %s", missing_fields_header)
    if missing_fields_items:
        logging.warning("Missing fields in iFDO items: %s", missing_fields_items)
    ifdo_data = {
        "image-set-header": image_set_header,
        "image-set-items": image_set_items,
    }
    errors = validate_ifdo(ifdo_data=ifdo_data)
    if errors:
        msg_error = "Validation errors in the output iFDO metadata file:\n"
        for error in errors:
            msg_error += f"{format_ifdo_validation_error(error['path'])}: {error['message']}\n"
        logging.warning(msg_error)
        # raise_value_error(f"Validation errors: {error_messages}")
    with Path(output_path).open("w") as file:
        json.dump(ifdo_data, file, indent=4)


def parse_ifdo_items(metadata: dict, ifdo_schema: dict) -> list:
    """Parse iFDO items from metadata.

    Args:
        metadata (dict): Metadata to parse.
        ifdo_schema (dict): iFDO schema.

    Returns:
        list: Parsed iFDO items.
    """
    ifdo_fields, required_fields, non_required_fields = get_ifdo_fields(ifdo_schema, "items")
    missing_fields = []
    ifdo_items = {}
    exif_data = []
    for _, image_metadata in metadata.iterrows():
        exif_data.append(map_exif_to_ifdo(image_metadata))
    exif_data = pd.DataFrame(exif_data)
    exif_data = exif_data.to_dict()
    if exif_data:
        for key in exif_data:
            if key not in metadata.columns:
                metadata[key] = list(exif_data[key].values())
    metadata = metadata.to_dict(orient="records")
    for item in metadata:
        missing_field = []
        ifdo_item = {}
        ifdo_item = map_fields_to_ifdo(item, ifdo_item, ifdo_fields, required_fields, missing_field, required=True)
        ifdo_item = map_fields_to_ifdo(item, ifdo_item, ifdo_fields, non_required_fields, missing_field, required=False)
        ifdo_items[item["image-filename"]] = ifdo_item
        missing_fields.append(missing_field)
    return ifdo_items, missing_fields


def parse_ifdo_header(dataset_metadata: dict, ifdo_schema: dict, metadata: pd.DataFrame) -> tuple:
    """Parse iFDO header from dataset metadata.

    Args:
        dataset_metadata (dict): Dataset metadata.
        ifdo_schema (dict): iFDO schema.
        metadata (pd.DataFrame): Metadata to parse.

    Returns:
        dict: Parsed iFDO header.
    """
    ifdo_fields, required_fields, non_required_fields = get_ifdo_fields(ifdo_schema, "header")
    missing_fields = []
    ifdo_header = {}
    if "output_path" in dataset_metadata:
        dataset_metadata["image-set-handle"] = str(dataset_metadata["output_path"])
    if "input_path" in dataset_metadata and "image-set-handle" not in dataset_metadata:
        dataset_metadata["image-set-handle"] = str(dataset_metadata["input_path"])
    if "image-datetime" not in dataset_metadata and "image-datetime" in metadata.columns:
        min_dt = metadata["image-datetime"].min()
        max_dt = metadata["image-datetime"].max()
        center_dt = min_dt + (max_dt - min_dt) / 2
        dataset_metadata["image-datetime"] = str(center_dt)

    if "image-latitude" not in dataset_metadata and "image-latitude" in metadata.columns:
        dataset_metadata["image-latitude"] = (metadata["image-latitude"].max() + metadata["image-latitude"].min()) / 2

    if "image-longitude" not in dataset_metadata and "image-longitude" in metadata.columns:
        dataset_metadata["image-longitude"] = (metadata["image-longitude"].max() + metadata["image-longitude"].min()) / 2

    ifdo_header = map_fields_to_ifdo(dataset_metadata, ifdo_header, ifdo_fields, required_fields, missing_fields, required=True)
    ifdo_header = map_fields_to_ifdo(dataset_metadata, ifdo_header, ifdo_fields, non_required_fields, missing_fields, required=False)
    return ifdo_header, missing_fields


def map_fields_to_ifdo(
    data: dict,
    ifdo_data: dict,
    schema: dict,
    fields: list,
    missing_fields: list,
    missing_fields_suffix: str = "",
    required: bool = False,
) -> dict:
    """Map fields from dataset metadata to iFDO header.

    Args:
        data (dict): Dataset metadata.
        ifdo_data (dict): iFDO data to populate.
        schema (dict): iFDO schema.
        fields (list): List of fields to map.
        missing_fields (list): List of missing fields.
        missing_fields_suffix (str): Suffix to append to missing fields.
        required (bool): Whether the fields are required.

    Returns:
        dict: Mapped iFDO header.
    """
    if not required and not fields:
        return data
    for field in fields:
        if field not in data:
            if not required:
                continue
            missing_field = f"{missing_fields_suffix}:{field}" if missing_fields_suffix else field
            missing_fields.append(missing_field)
        schema_field = schema[field]
        if schema_field.get("type") == "string":
            ifdo_data[field] = data.get(field, schema_field.get("description", ""))
        elif schema_field.get("type") == "object":
            ifdo_data[field] = {}
            sub_fields = schema_field.get("required", []) if required else schema_field.get("properties", {}).keys()
            ifdo_data[field] = map_fields_to_ifdo(
                data.get(field, schema_field.get(field, {})),
                ifdo_data[field],
                schema_field.get("properties", {}),
                sub_fields,
                missing_fields,
                field,
                required,
            )
        elif schema_field.get("type") == "array":
            ifdo_data[field] = data.get(field, [schema_field.get("description", "")])
        elif schema_field.get("type") == "number":
            ifdo_data[field] = data.get(field, schema_field.get("description", ""))
        else:
            ifdo_data[field] = data.get(field, schema_field.get("description", ""))
    return ifdo_data


def map_exif_to_ifdo(metadata: dict) -> str | None:
    """Map EXIF metadata to iFDO format.

    Args:
        metadata (dict): Metadata to convert.

    Returns:
        str | None: Converted metadata in iFDO format.
    """

    def clean_dict(dict_to_clean: dict) -> dict:
        """Recursively remove keys with None, empty strings, or empty dictionaries.

        Args:
            dict_to_clean (dict): Dictionary to clean.

        Returns:
            dict: Cleaned dictionary.
        """
        if isinstance(dict_to_clean, dict):
            return {k: clean_dict(v) for k, v in dict_to_clean.items() if v not in (None, "", {}) and clean_dict(v) not in (None, "", {}, [])}
        return dict_to_clean

    try:
        image_latitude = metadata.get("GPSLatitude") * (1 if metadata.get("GPSLatitudeRef") == "N" else -1)
    except TypeError:
        image_latitude = ""
    try:
        image_longitude = metadata.get("GPSLongitude") * (1 if metadata.get("GPSLongitudeRef") == "E" else -1)
    except TypeError:
        image_longitude = ""
    try:
        image_altitude = metadata.get("GPSAltitude") * (1 if metadata.get("GPSAltitudeRef") == "0" else -1)
    except TypeError:
        image_altitude = ""

    exif_to_ifdo = {
        "image-latitude": image_latitude,
        "image-longitude": image_longitude,
        "image-sensor": {
            "name": (metadata.get("Make", "") + " " + metadata.get("Model", "")).strip(),
        },
        "image-acquisition-settings": {
            "make": metadata.get("Make", ""),
            "model": metadata.get("Model", ""),
            "software": metadata.get("Software", ""),
            "exposure_time": metadata.get("ExposureTime", ""),
            "f_number": metadata.get("FNumber", ""),
            "iso_speed": metadata.get("ISOSpeedRatings", ""),
            "focal_length": metadata.get("FocalLength", ""),
            "orientation": metadata.get("Orientation", ""),
            "flash": metadata.get("Flash", ""),
            "white_balance": metadata.get("WhiteBalance", ""),
            "metering_mode": metadata.get("MeteringMode", ""),
            "scene_capture_type": metadata.get("SceneCaptureType", ""),
        },
        "image-datetime": metadata.get("DateTimeOriginal", metadata.get("CreateDate", "")),
        "image-altitude-meters": image_altitude,
        "image-abstract": metadata.get("ImageDescription", ""),
        "image-creator": {"name": metadata.get("Artist", "")},
        "image-copyright": metadata.get("Copyright", ""),
    }

    cleaned = clean_dict(exif_to_ifdo)
    return cleaned if cleaned else {}


def format_ifdo_validation_error(text: list) -> str:
    """Format error message.

    Args:
        text (list): List of error messages.

    Returns:
        str: Formatted error message.
    """
    if len(text) > 3:  # noqa: PLR2004
        return f"...{'.'.join(map(str, text[-3:]))}"
    return ".".join(map(str, text))


def parse_validation_errors(errors: list, schema: dict) -> list:
    """Parse validation errors.

    Args:
        errors (list): List of validation errors.
        schema (dict): JSON schema.

    Returns:
        list: Parsed validation errors.
    """
    ifdo_fields, required_fields, _ = get_ifdo_fields(schema, "items")
    image_uuid_pattern = schema["$defs"]["uuid"]["pattern"]
    default_error_message = "is not valid under any of the given schemas"
    for idx, error in enumerate(errors):
        if default_error_message in error["message"]:
            error_message = error["message"].split(default_error_message)[0].strip()
            error_message = ast.literal_eval(error_message)
            missing = [field for field in required_fields if field not in error_message]
            additional = [item for item in error_message if item not in ifdo_fields]
            is_valid_uuid = bool(re.match(image_uuid_pattern, error_message.get("image-uuid", "")))
            output_error_message = ""
            if missing:
                output_error_message += f"Missing fields: {missing}. "
            if additional:
                output_error_message += f"Additional fields (this will not make the metadata invalid): {additional}. "
            if not is_valid_uuid:
                output_error_message += f"Invalid or missing 'image-uuid' value: '{error_message.get('image-uuid', '')}'. "
            if not output_error_message:
                output_error_message = error["message"]
            errors[idx]["message"] = output_error_message.strip()
    return errors


def get_ifdo_fields(schema: dict, section: str) -> tuple:
    """Get required fields from iFDO schema.

    Args:
        schema (dict): JSON schema.
        section (str): Section of the schema to get fields from.

    Returns:
        tuple: iFDO fields, required fields, non-required fields.
    """
    required_fields = schema["$defs"]["image-item-core"]["required"] if section == "items" else schema["properties"]["image-set-header"]["required"]
    ifdo_fields = schema["$defs"]["iFDO-fields"]["properties"]
    if section == "items":
        excluding_items = ["image-set-name", "image-set-handle", "image-set-ifdo-version", "image-set-uuid"]
        ifdo_item_fields = [field for field in ifdo_fields if field not in excluding_items]
    else:
        ifdo_item_fields = ifdo_fields.keys()
    non_required_fields = set(ifdo_item_fields) - set(required_fields)
    return ifdo_fields, required_fields, non_required_fields
