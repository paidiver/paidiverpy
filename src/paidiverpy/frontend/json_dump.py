"""This module provides functions to extract values from a Panel layout and convert them into a structured JSON-like dictionary."""

import re
import panel as pn


def find_deep_layout(layout: pn.widgets.Widget, founds: list[pn.widgets.Widget]) -> list[pn.widgets.Widget]:
    """Recursively find all widgets and layouts in a Panel layout.

    Args:
        layout (pn.widgets.Widget): The Panel layout or widget to search.
        founds (list[pn.widgets.Widget]): A list to collect found widgets and layouts.

    Returns:
        list[pn.widgets.Widget]: A list of found widgets and layouts.
    """
    if isinstance(layout, pn.param.ParamFunction):
        inner = getattr(layout, "_inner_layout", None)
        layout = inner if inner is not None else layout.object()
    if isinstance(layout, pn.param.ParamFunction | pn.widgets.Widget):
        founds.append(layout)
    elif isinstance(layout, pn.Column | pn.Row):
        for child in layout:
            find_deep_layout(child, founds)
    return founds


def check_valid_inputs(widget: pn.widgets.Widget, step: bool = False) -> bool:
    """Check if a widget is valid for extraction.

    Args:
        widget (pn.widgets.Widget): The widget to check.
        step (bool): If True, additional checks for step widgets are applied.

    Returns:
        bool: True if the widget is valid, False otherwise.
    """
    if isinstance(widget, pn.widgets.Button):
        return False
    if not hasattr(widget, "name") or not hasattr(widget, "value"):
        return False
    if hasattr(widget, "disabled") and widget.disabled:
        return False
    if "Steps" in widget.name:
        return False
    if "type_selector" in widget.name and not step:
        return False
    return "Provide" not in widget.name


def parse_name(name: str) -> list:
    """Parse a widget name into a list of keys.

    Args:
        name (str): The name of the widget, which may contain dots and brackets.

    Returns:
        list: A list of keys parsed from the name, converting numeric parts to integers.
    """
    return [int(part) if part.isdigit() else part for part in re.findall(r"\w+|\[\d+\]", name.replace("[", ".").replace("]", ""))]


def insert_nested(result: dict, keys: list, value: any) -> None:
    """Insert a value into a nested dictionary structure based on keys.

    Args:
        result (dict): The dictionary to insert into.
        keys (list): A list of keys indicating the path to insert the value.
        value (any): The value to insert.
    """
    current = result
    for i, key in enumerate(keys):
        if isinstance(key, int):
            while len(current) <= key:
                current.append({})
            current = current[key]
        elif i < len(keys) - 1:
            if key not in current:
                current[key] = [] if isinstance(keys[i + 1], int) else {}
            current = current[key]
        else:
            current[key] = value


def extract_values(widgets: list[pn.widgets.Widget], step: bool = False) -> dict:
    """Extract values from a list of widgets and return them as a structured dictionary.

    Args:
        widgets (list[pn.widgets.Widget]): A list of Panel widgets to extract values from.
        step (bool): If True, the extraction will consider step-specific widgets.

    Returns:
        dict: A dictionary containing the extracted values, structured by widget names.
    """
    result = {}
    list_collections = {}

    for widget in widgets:
        if not check_valid_inputs(widget, step):
            continue
        if "type_selector" in widget.name and step:
            step = False
        keys = parse_name(widget.name)
        value = widget.value

        if any(isinstance(k, int) for k in keys):
            list_name = keys[0]
            list_index = keys[1]
            rest_keys = keys[2:]

            if list_name not in list_collections:
                list_collections[list_name] = {}

            if list_index not in list_collections[list_name]:
                list_collections[list_name][list_index] = {}

            insert_nested(list_collections[list_name][list_index], rest_keys, value)
        else:
            insert_nested(result, keys, value)

    for list_name, items in list_collections.items():
        indexed_items = [items[i] for i in sorted(items) if items[i]]
        result[list_name] = indexed_items

    return result


def extract_json(layout: pn.widgets.Widget, step: bool = False) -> dict:
    """Extract JSON-like dictionary from a Panel layout or widget.

    Args:
        layout (pn.widgets.Widget): The Panel layout or widget to extract from.
        step (bool): If True, the extraction will consider step-specific widgets.

    Returns:
        dict: A dictionary containing the extracted values, structured by widget names.
    """
    inputs = []
    find_deep_layout(layout, inputs)
    json = extract_values(inputs, step)
    if step:
        step_name = json["steps"][0]["type_selector"].replace("Config", "").lower()
        step_params = json["steps"][0].copy()
        del step_params["type_selector"]
        json = {step_name: step_params}
    return json
