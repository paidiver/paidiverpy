from pathlib import Path
from typing import get_args
from typing import get_origin
from pydantic_core import PydanticUndefinedType

# from paidiverpy.config.colour_params import COLOUR_LAYER_METHODS
# from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
# from paidiverpy.config.position_params import POSITION_LAYER_METHODS
# from paidiverpy.config.sampling_params import SAMPLING_LAYER_METHODS
# from paidiverpy.config.step_config import StepConfigUnion


def define_default_value(field):
    if field.default_factory is not None:
        return field.default_factory() if callable(field.default_factory) else field.default_factory
    if type(field.default) is PydanticUndefinedType:
        return None
    return field.default if field.default is not None else None

def parse_default_params(model, steps=False):
    params = parse_fields_from_pydantic_model(model)
    if "name" in params and not steps:
        del params["name"]
    if "step_name" in params:
        del params["step_name"]
    if "test" in params and not steps:
        del params["test"]
    if "input_path" in params and not steps:
        input_path = {}
        input_path["type"] = "str"
        input_path["default"] = ""
        input_path["description"] = params["input_path"]["description"]
        params["input_path"] = input_path
    if "metadata_path" in params and not steps:
        metadata_path = {}
        metadata_path["type"] = "str"
        metadata_path["default"] = ""
        metadata_path["description"] = params["metadata_path"]["description"]
        params["metadata_path"] = metadata_path
    if "metadata_type" in params and not steps:
        metadata_type = {}
        metadata_type["type"] = "literal"
        metadata_type["default"] = None
        metadata_type["options"] = params["metadata_type"]["field_options"]["literal"]
        metadata_type["description"] = params["metadata_type"]["description"]
        params["metadata_type"] = metadata_type

    return params


def parse_get_origin_field(field, origin_field):
    output = {}
    output["default"] = define_default_value(field)
    output["description"] = field.description if field.description else ""
    if origin_field.__name__ in ["Literal", "LiteralType", "literal"]:
        output["type"] = "literal"
        output["options"] = list(get_args(field.annotation))
    elif origin_field.__name__ in ["UnionType", "Union"]:
        arguments = get_args(field.annotation)
        if len(arguments) == 2 and Path in arguments:
            output["type"] = "str"
            output["default"] = output["default"] if output["default"] is not None else ""
        else:
            output["type"] = "union"
            output["field_options"] = {}
            for argument in get_args(field.annotation):
                if argument is Path:
                    continue  # Path is not supported in frontend
                if argument is None:
                    output["field_options"]["None"] = "null"
                elif argument.__name__ in ["Literal", "LiteralType", "literal"]:
                    output["field_options"]["literal"] = list(get_args(argument))
                elif argument.__name__ == "list":
                    output["field_options"]["list"] = get_args(argument)[0]
                elif hasattr(argument, "model_json_schema"):
                    output["field_options"][argument.__name__] = argument.model_json_schema()["description"]
                else:
                    output["field_options"][argument.__name__] = argument.__name__

    elif origin_field.__name__ == "list":
        output["type"] = "list"
        output["item_type"] = get_args(field.annotation)[0] if get_args(field.annotation) else "str"
        output["item_default"] = define_default_value(field)
    else:
        output["type"] = origin_field.__name__ if origin_field else "Any"
        if hasattr(origin_field, "__name__"):
            output["type"] = origin_field.__name__
        elif hasattr(origin_field, "__origin__"):
            output["type"] = origin_field.__origin__.__name__
    return output


def parse_fields_from_pydantic_model(model):
    output = {}
    for name, field in model.model_fields.items():
        default_value = define_default_value(field)
        output[name] = {
            "default": default_value,
            "description": field.description
        }
        origin_field = get_origin(field.annotation)
        if not origin_field:
            field_type = field.annotation.__name__ if field.annotation else "Any"
            output[name]["type"] = field_type
            continue
        output[name] = parse_get_origin_field(field, origin_field)
    return output
