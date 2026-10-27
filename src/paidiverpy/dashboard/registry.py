"""Convert Paidiverpy's live Pydantic models into dashboard form metadata."""

# Field values are intentionally heterogeneous because they are serialized to JSON.
# ruff: noqa: ANN401, TC002

from __future__ import annotations
from typing import Any
from typing import Literal
from typing import get_args
from typing import get_origin
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from paidiverpy.models.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.models.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.models.position_params import POSITION_LAYER_METHODS
from paidiverpy.models.sampling_params import SAMPLING_LAYER_METHODS
from paidiverpy.models.step_config import CustomConfig

METHODS = {
    "position": POSITION_LAYER_METHODS,
    "sampling": SAMPLING_LAYER_METHODS,
    "convert": CONVERT_LAYER_METHODS,
    "colour": COLOUR_LAYER_METHODS,
}


def _json_value(value: Any) -> Any:
    if value is PydanticUndefined:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return value


def _field(name: str, field: FieldInfo, *, required: bool | None = None) -> dict[str, Any]:
    annotation = field.annotation
    args = list(get_args(annotation))
    literal = annotation if get_origin(annotation) is Literal else next((arg for arg in args if get_origin(arg) is Literal), None)
    options = list(get_args(literal)) if literal is not None else []
    nullable = type(None) in args
    type_name = "json"
    candidates = [annotation, *args]
    if bool in candidates:
        type_name = "boolean"
    elif int in candidates:
        type_name = "integer"
    elif float in candidates:
        type_name = "number"
    elif str in candidates:
        type_name = "string"
    default = field.default_factory() if field.default_factory else _json_value(field.default)
    return {
        "name": name,
        "type": "select" if options else type_name,
        "options": options,
        "nullable": nullable,
        "default": default,
        "required": field.is_required() if required is None else required,
        "description": field.description or "",
    }


def _model_fields(model: type, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    return [_field(name, field) for name, field in model.model_fields.items() if name not in excluded]


def dashboard_registry() -> dict[str, Any]:
    """Return current processing modes and their parameter definitions."""
    categories = []
    for category, methods in METHODS.items():
        modes = []
        for mode, definition in methods.items():
            model = definition["params"]
            modes.append(
                {
                    "name": mode,
                    "label": mode.replace("_", " ").title(),
                    "description": (model.__doc__ or "").strip().split("\n", 1)[0],
                    "fields": _model_fields(model),
                }
            )
        categories.append({"name": category, "label": category.title(), "modes": modes})
    categories.append(
        {
            "name": "custom",
            "label": "Custom",
            "modes": [
                {
                    "name": "custom",
                    "label": "Custom algorithm",
                    "description": "Run an algorithm defined in a user-provided Python file.",
                    "fields": _model_fields(CustomConfig, {"name", "step_name", "test"}),
                }
            ],
        }
    )
    general_excluded = {"name", "step_name", "sampling", "convert"}
    general_steps = [category for category in categories if category["name"] in {"sampling", "convert"}]
    return {
        "general": _model_fields(GeneralConfig, general_excluded),
        "general_steps": general_steps,
        "categories": categories,
    }
