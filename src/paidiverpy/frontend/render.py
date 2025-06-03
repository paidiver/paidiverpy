"""WidgetRenderer class for rendering widgets based on configuration parameters."""

import types
from typing import Literal
from typing import get_args
import panel as pn
from paidiverpy.frontend.parse import define_default_value
from paidiverpy.frontend.parse import parse_default_params
from paidiverpy.models.client_params import ClientParams  # noqa: F401
from paidiverpy.models.colour_params import *  # noqa: F403
from paidiverpy.models.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.models.convert_params import *  # noqa: F403
from paidiverpy.models.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.models.custom_params import *  # noqa: F403
from paidiverpy.models.general_config import GeneralConfig  # noqa: F401
from paidiverpy.models.open_params import *  # noqa: F403
from paidiverpy.models.position_params import *  # noqa: F403
from paidiverpy.models.position_params import POSITION_LAYER_METHODS
from paidiverpy.models.sampling_params import *  # noqa: F403
from paidiverpy.models.sampling_params import SAMPLING_LAYER_METHODS
from paidiverpy.models.step_config import ColourConfig  # noqa: F401
from paidiverpy.models.step_config import ConvertConfig  # noqa: F401
from paidiverpy.models.step_config import CustomConfig  # noqa: F401
from paidiverpy.models.step_config import PositionConfig  # noqa: F401
from paidiverpy.models.step_config import SamplingConfig  # noqa: F401
from paidiverpy.utils.base_model import BaseModel

OPTIONAL = 2


class WidgetRenderer:
    """Class for rendering widgets based on configuration parameters.

    Args:
        steps (bool): If True, the renderer will handle step-specific widgets.
        step_parameters (dict, optional): Parameters for the step if applicable.
    """

    def __init__(self, steps: bool = False, step_parameters: BaseModel | None = None):
        self.steps = steps
        if step_parameters is not None:
            self.step_class = step_parameters.__class__.__name__
            self.step_parameters = step_parameters.to_dict()
        else:
            self.step_class = None
            self.step_parameters = None

    def create_widget(self, name: str, field: dict, html_h_tag: int = 2) -> pn.Column:
        """Create a widget based on the field type and name.

        Args:
            name (str): The name of the field.
            field (dict): The field definition containing type, description, default value, etc.
            html_h_tag (int): The HTML heading tag level for the title.

        Returns:
            pn.Column: A Panel Column containing the title and the input widget.
        """
        model_class = globals().get(field["type"])
        if model_class and hasattr(model_class, "model_config") and model_class.model_config.get("extra") == "allow":
            default = {}
            for local_name, local_field in model_class.model_fields.items():
                default[local_name] = define_default_value(local_field)
            field = {"default": default, "description": model_class.__doc__, "type": "dict"}
        type_ = field["type"]
        description = field.get("description", "")
        default = field.get("default")
        name_title = name.replace("_", " ").capitalize()
        html_pane = pn.pane.HTML(
            f"<div class='ppy-pn-w-50'><div class='ppy-pn-title-{html_h_tag} ppy-pn-bold'>{name_title}</div>"
            f"<div class='ppy-pn-description'>{description}</div></div>"
        )

        input_widget = self.get_input_widget(type_.lower(), field, name, default, html_h_tag)
        if input_widget:
            return pn.Column(
                html_pane,
                input_widget,
                styles={"padding": "0px"},
            )

        if not input_widget and type_.lower() == "union":
            options = list(field["field_options"].keys())

            if len(options) == OPTIONAL and "NoneType" in options:
                input_widget = self.create_optional_widget(name, field, options, default, html_h_tag=html_h_tag)
            else:
                if "NoneType" in options:
                    options[options.index("NoneType")] = "Not provided (NoneType)"
                    if default is None:
                        default_type = "Not provided (NoneType)"
                else:
                    default_type = next((k for k in options if k.lower() in str(type(default)).lower()), options[0])

                if self.step_parameters and "steps[" in name:
                    selector = pn.widgets.Select(name=f"{name} type_selector", options=options, value=self.step_class)
                else:
                    selector = pn.widgets.Select(name=f"{name} type_selector", options=options, value=default_type)

                input_widget = pn.Column(
                    selector,
                    pn.bind(self.render_union_input, selector, field=field, name=name, default=default, html_h_tag=html_h_tag),
                )
        if input_widget:
            return pn.Column(html_pane, input_widget, styles={"padding": "0px"})
        return self.render_custom_types(model_class=type_, prefix=name, html_h_tag=html_h_tag)

    def create_optional_widget(
        self, name: str, field: dict, options: list[str], default: str | float | bool | None = None, html_h_tag: int = 2
    ) -> pn.Column:
        """Create a widget for an optional field.

        Args:
            name (str): The name of the field.
            field (dict): The field definition containing type, description, default value, etc.
            options (list[str]): The options for the union type.
            default (str | float | bool | None): The default value for the field.
            html_h_tag (int): The HTML heading tag level for the title.

        Returns:
            pn.Column: A Panel Column containing the title and the input widget.
        """
        other_type = next(opt for opt in options if opt != "NoneType")

        if self.step_parameters:
            new_names = name.split(".")[1:]
            value = self.step_parameters.copy()
            for new_name in new_names:
                value = value.get(new_name)
            if value is not None:
                default = True

        provide_checkbox = pn.widgets.Checkbox(name=f"Provide {name}?", value=default is not None)

        return pn.Column(
            provide_checkbox,
            pn.bind(self.render_union_input, other_type, field=field, name=name, default=default, provide=provide_checkbox, html_h_tag=html_h_tag),
        )

    def render_custom_types(self, model_class: str, prefix: str | None = None, html_h_tag: int = 2) -> pn.Column:
        """Render custom types based on the model class and field definition.

        Args:
            model_class (str): The name of the model class.
            field (dict, optional): The field definition if applicable.
            prefix (str, optional): The prefix for the field name.
            html_h_tag (int): The HTML heading tag level for the title.

        Returns:
            pn.Column: A Panel Column containing the rendered widgets.
        """
        model_class = globals().get(model_class)

        field_meta = parse_default_params(model_class, self.steps)
        if "mode" in field_meta and "params" in field_meta:
            new_field_meta = {k: v for k, v in field_meta.items() if k not in ["mode", "params"]}
            widgets = []
            if new_field_meta:
                html_h_tag = html_h_tag + 1
                for field_name, field in new_field_meta.items():
                    full_name = f"{prefix}.{field_name}" if prefix else field_name
                    widget = self.create_widget(full_name, field, html_h_tag=html_h_tag)
                    widgets.append(widget)
            widgets = self.render_method_with_mode_params(field_meta, model_class, prefix, html_h_tag, widgets=widgets)
        else:
            widgets = []
            html_h_tag = html_h_tag + 1
            for field_name, field in field_meta.items():
                full_name = f"{prefix}.{field_name}" if prefix else field_name
                widget = self.create_widget(full_name, field, html_h_tag=html_h_tag)
                widgets.append(widget)

        return pn.Column(*widgets, styles={"padding": "0px"})

    def render_list_input(self, field: dict, name: str, html_h_tag: int = 2) -> pn.Column:
        """Render a list input widget based on the field definition.

        Args:
            field (dict): The field definition containing type, description, default value, etc.
            name (str): The name of the field.
            html_h_tag (int): The HTML heading tag level for the title.

        Returns:
            pn.Column: A Panel Column containing the list input widget.
        """
        item_type = field.get("item_type", "str") if "field_options" not in field else field["field_options"]["list"]
        new_html_h_tag = html_h_tag + 1

        inputs = []

        inputs_container = pn.Column()

        def create_list_widget(index: int) -> tuple[pn.pane.HTML, dict]:
            html_pane = pn.pane.HTML(f"<div class='ppy-pn-title-{html_h_tag} ppy-pn-italic'>ITEM {index + 1}</div>")
            if isinstance(item_type, types.UnionType):
                field_def = {"type": "union", "field_options": {}}
                for item in item_type.__args__:
                    field_def["field_options"][item.__name__] = item.model_json_schema()["description"]
            else:
                field_def = {"type": item_type.__name__}
            widget = self.create_widget(f"{name}[{index + 1}]", field_def, html_h_tag=new_html_h_tag)
            return html_pane, widget

        def create_row(index: int, widget: dict) -> pn.Row:
            remove_btn = pn.widgets.Button(name="Remove", button_type="danger", width=80)

            def remove_item() -> None:
                inputs.pop(index)
                inputs_container[:] = [create_row(i, item) for i, item in enumerate(inputs)]

            remove_btn.on_click(remove_item)

            if self.steps:
                return pn.Row(
                    widget["widget"],
                )
            return pn.Row(pn.Column(widget["html"], widget["widget"]), remove_btn)

        def add_item() -> None:
            html, widget = create_list_widget(len(inputs))
            if self.steps:
                inputs.append({"widget": widget})
            else:
                inputs.append({"html": html, "widget": widget})
            inputs_container[:] = [create_row(i, item) for i, item in enumerate(inputs)]

        add_item()

        if self.steps:
            return pn.Column(
                pn.layout.Divider(),
                inputs_container,
                pn.layout.Divider(),
            )
        add_button_top = pn.widgets.Button(name="Add Item", button_type="primary")
        add_button_bottom = pn.widgets.Button(name="Add Item", button_type="primary")
        add_button_top.on_click(add_item)
        add_button_bottom.on_click(add_item)

        return pn.Column(
            pn.Row(add_button_top),
            pn.layout.Divider(),
            inputs_container,
            pn.layout.Divider(),
            pn.Row(add_button_bottom),
        )

    def render_union_input(
        self, selected_type: str, field: dict, name: str, default: str | float | bool | None = None, provide: bool = True, html_h_tag: int = 2
    ) -> pn.widgets.Widget:
        """Render the input widget for a union type field.

        Args:
            selected_type (str): The selected type from the union.
            field (dict): The field definition containing type, description, default value, etc.
            name (str): The name of the field.
            default (str | float | bool | None): The default value for the field.
            provide (bool): Whether to provide the input widget or not.
            html_h_tag (int): The HTML heading tag level for the title.

        Returns:
            pn.widgets.Widget: The input widget for the selected type.
        """
        html_h_tag = html_h_tag + 1
        input_widget = self.get_input_widget(selected_type, field, name, default, html_h_tag, provide=provide)
        if not input_widget:
            input_widget = self.render_custom_types(model_class=selected_type, prefix=name, html_h_tag=html_h_tag)
        return input_widget

    def _get_default_from_step_parameters(self, default: str | float | bool | None, name: str) -> str | float | bool | None:
        if self.step_parameters:
            new_names = name.split(".")[1:]
            value = self.step_parameters.copy()
            for new_name in new_names:
                value = value.get(new_name)
            if value is not None:
                default = value
        return default

    def get_input_widget(  # noqa: C901
        self,
        type_: str,
        field: dict,
        name: str,
        default: str | float | bool | None = None,
        html_h_tag: int = 2,
        provide: bool = True,
    ) -> pn.widgets.Widget | None:
        """Get the input widget based on the type and field definition.

        Args:
            type_ (str): The type of the field.
            field (dict): The field definition containing type, description, default value, etc.
            name (str): The name of the field.
            default (str | float | bool | None): The default value for the field.
            html_h_tag (int): The HTML heading tag level for the title.
            provide (bool): Whether to provide the input widget or not.

        Returns:
            pn.widgets.Widget | None: The input widget for the field, or None if not applicable.
        """
        result = None
        selected_type = type_.lower()
        default = self._get_default_from_step_parameters(default, name)
        if not provide:
            result = pn.widgets.TextInput(value="", disabled=True)
        elif selected_type == "int":
            result = pn.widgets.IntInput(name=name, value=default)
        elif selected_type == "float":
            result = pn.widgets.FloatInput(name=name, value=default)
        elif selected_type == "bool":
            result = pn.widgets.Checkbox(name=name, value=default)
        elif selected_type == "tuple":
            default = (1, 2) if default is None else tuple(default) if isinstance(default, list) else default
            result = pn.widgets.TextInput(name=name, value=str(default), placeholder="(item1, item2, ...)")
        elif selected_type == "dict":
            default = {"key": "value"} if default is None or default == {} else default
            result = pn.widgets.JSONEditor(name=name, value=default, width=400, mode="tree")
        elif selected_type == "str":
            result = pn.widgets.TextInput(name=name, value=default)
        elif selected_type == "not provided (nonetype)":
            result = pn.pane.HTML("<div class='ppy-pn-description ppy-pn-italic'>No input needed</div>")
        elif selected_type in ["literal"]:
            opts = field["field_options"].get("literal", []) if "field_options" in field else field.get("options", [])
            result = pn.widgets.Select(name=name, options=opts, value=default)
        elif selected_type == "list":
            item_type = field.get("item_type", "str")
            item_default = field.get("item_default", "")
            result = self.render_list_input(field, name, html_h_tag=html_h_tag) if item_type else pn.widgets.TextInput(name=name, value=item_default)
        return result

    def render_method_with_mode_params(
        self, field_meta: dict, model_class: str, prefix: str | None = None, html_h_tag: int = 2, widgets: list[pn.Column] | None = None
    ) -> list[pn.Column]:
        """Render the method with mode and parameters based on the field metadata.

        Args:
            field_meta (dict): The field metadata containing mode and parameters.
            model_class (str): The name of the model class.
            prefix (str | None): The prefix for the field name.
            html_h_tag (int): The HTML heading tag level for the title.
            widgets (list[pn.Column] | None): Existing widgets to append to.

        Returns:
            list[pn.Column]: A list of Panel Columns containing the rendered widgets.
        """
        mode_options = field_meta["mode"]["options"]

        mode_select = pn.widgets.Select(
            name=f"{prefix}.mode",
            options=mode_options,
            value=mode_options[0],
        )

        def get_param_model(model_class: str) -> dict:
            mapping = {
                "SamplingConfig": SAMPLING_LAYER_METHODS,
                "PositionConfig": POSITION_LAYER_METHODS,
                "ColourConfig": COLOUR_LAYER_METHODS,
                "ConvertConfig": CONVERT_LAYER_METHODS,
            }
            return mapping.get(model_class)

        def make_params_widget(selected_mode: str) -> pn.Column:
            param_model_name = get_param_model(model_class.__name__)[selected_mode]["params"]
            if param_model_name:
                return self.render_custom_types(model_class=param_model_name.__name__, prefix=f"{prefix}.params", html_h_tag=html_h_tag + 1)
            return pn.pane.Markdown(f"*No parameters available for mode `{selected_mode}`*")

        params_widget = pn.bind(make_params_widget, mode_select)
        if widgets is None:
            widgets = []

        widgets.extend(
            [
                pn.Column(
                    pn.pane.HTML(
                        f"<div class='ppy-pn-title-{html_h_tag} ppy-pn-bold'>Mode</div>"
                        f"<div class='ppy-pn-description'>{field_meta['mode'].get('description', '')}</div>"
                    ),
                    mode_select,
                    pn.pane.HTML(
                        f"<div class='ppy-pn-title-{html_h_tag} ppy-pn-bold'>Params</div>"
                        f"<div class='ppy-pn-description'>{field_meta['params'].get('description', '')}</div>"
                    ),
                    params_widget,
                )
            ]
        )
        return widgets

    def widget_from_literal(self, field_name: str, literal_type: Literal["option1", "option2", "option3"]) -> pn.widgets.Select:
        """Create a widget from a literal type.

        Args:
            field_name (str): The name of the field.
            literal_type (Literal): The literal type to create the widget from.

        Returns:
            pn.widgets.Select: A Panel Select widget with options from the literal type.
        """
        options = list(get_args(literal_type))
        return pn.widgets.Select(name=field_name, options=options, value=options[0])
