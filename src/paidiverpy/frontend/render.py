import types
from typing import Literal
from typing import get_args
import panel as pn
from paidiverpy.config.client_params import ClientParams
from paidiverpy.config.colour_params import *
from paidiverpy.config.colour_params import COLOUR_LAYER_METHODS
from paidiverpy.config.convert_params import *
from paidiverpy.config.convert_params import CONVERT_LAYER_METHODS
from paidiverpy.config.custom_params import *
from paidiverpy.config.general_config import GeneralConfig
from paidiverpy.config.open_params import *
from paidiverpy.config.position_params import *
from paidiverpy.config.position_params import POSITION_LAYER_METHODS
from paidiverpy.config.sampling_params import *
from paidiverpy.config.sampling_params import SAMPLING_LAYER_METHODS
from paidiverpy.config.step_config import ColourConfig
from paidiverpy.config.step_config import ConvertConfig
from paidiverpy.config.step_config import PositionConfig
from paidiverpy.config.step_config import SamplingConfig
from paidiverpy.frontend.parse import parse_default_params


class WidgetRenderer:
    def __init__(self, steps=False, step_parameters=None):
        self.steps = steps
        if step_parameters is not None:
            self.step_class = step_parameters.__class__.__name__
            self.step_parameters = step_parameters.to_dict()
        else:
            self.step_class = None
            self.step_parameters = None


    def create_widget(self, name, field, html_h_tag=2, is_list=False):
        type_ = field["type"]
        description = field.get("description", "")
        default = field.get("default", None)
        name_title = name.replace("_", " ").capitalize()
        html_pane = pn.pane.HTML(
            f"<div class='ppy-pn-w-50'><div class='ppy-pn-title-{html_h_tag} ppy-pn-bold'>{name_title}</div>"
            f"<div class='ppy-pn-description'>{description}</div></div>"
        )

        input_widget = self.get_input_widget(type_.lower(), field, name, default, html_h_tag)
        if input_widget:
            return pn.Column(
                html_pane,
                input_widget, styles={"padding": "0px"},
            )

        if not input_widget and type_.lower() == "union":
            options = list(field["field_options"].keys())

            if len(options) == 2 and "NoneType" in options:
                other_type = next(opt for opt in options if opt != "NoneType")

                if self.step_parameters:
                    new_names = name.split(".")[1:]
                    value = self.step_parameters.copy()
                    for new_name in new_names:
                        value = value.get(new_name)
                    if value is not None:
                        default = True

                provide_checkbox = pn.widgets.Checkbox(name=f"Provide {name}?", value=default is not None)

                input_widget = pn.Column(
                    provide_checkbox,
                    pn.bind(
                        self.render_union_input,
                        other_type,
                        field=field,
                        name=name,
                        default=default,
                        provide=provide_checkbox,
                        html_h_tag=html_h_tag
                    )
                )
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
            return pn.Column(
                html_pane,
                input_widget, styles={"padding": "0px"}
            )
        return self.render_custom_types(model_class=type_, field=field, prefix=name, default_values=default, html_h_tag=html_h_tag)

    # return pn.pane.Markdown(f"*No widget for type `{type_}`*")


    def render_custom_types(self, model_class, field=None, prefix=None, default_values=None, html_h_tag=2):
        model_class = globals().get(model_class)

        field_meta = parse_default_params(model_class, self.steps)
        if "mode" in field_meta and "params" in field_meta:
            #create a new field_meta without mode and params
            new_field_meta = {k: v for k, v in field_meta.items() if k not in ["mode", "params"]}
            widgets = []
            if new_field_meta:
                html_h_tag = html_h_tag + 1
                for field_name, field in new_field_meta.items():
                    full_name = f"{prefix}.{field_name}" if prefix else field_name
                    widget = self.create_widget(full_name, field , html_h_tag=html_h_tag)
                    widgets.append(widget)
            widgets = self.render_method_with_mode_params(field_meta,
                                                    model_class,
                                                    prefix,
                                                    html_h_tag,
                                                    widgets=widgets)
        else:
            widgets = []
            html_h_tag = html_h_tag + 1
            for field_name, field in field_meta.items():
                # if prefix and "." in prefix:
                #     prefix = prefix.split(".")[-1]
                full_name = f"{prefix}.{field_name}" if prefix else field_name
                widget = self.create_widget(full_name, field , html_h_tag=html_h_tag)
                widgets.append(widget)

        return pn.Column(*widgets, styles={"padding": "0px"})

    def render_list_input(self, field, name, html_h_tag=2):
        item_type = field.get("item_type", "str") if "field_options" not in field else field["field_options"]["list"]
        new_html_h_tag = html_h_tag + 1

        inputs = []  # List of widget objects (each item is a dict with {widget, row})

        inputs_container = pn.Column()

        def create_list_widget(index):

            html_pane = pn.pane.HTML(
                f"<div class='ppy-pn-title-{html_h_tag} ppy-pn-italic'>ITEM {index + 1}</div>"
            )

            # Define the item field type
            if isinstance(item_type, types.UnionType):
                field_def = {"type": "union", "field_options": {}}
                for item in item_type.__args__:
                    field_def["field_options"][item.__name__] = item.model_json_schema()["description"]
            else:
                field_def = {"type": item_type.__name__}


            widget = self.create_widget(f"{name}[{index + 1}]", field_def, html_h_tag=new_html_h_tag, is_list=True)
            return html_pane, widget



        def create_row(index, widget):
            remove_btn = pn.widgets.Button(name="Remove", button_type="danger", width=80)

            def remove_item(event=None):
                inputs.pop(index)
                update_inputs()

            remove_btn.on_click(remove_item)

            if self.steps:
                return pn.Row(
                    widget["widget"],
                )
            return pn.Row(
                pn.Column(widget["html"], widget["widget"]),
                remove_btn
            )

        def update_inputs():
            # Rebuild the UI using existing widgets, reindex rows
            inputs_container[:] = [
                create_row(i, item) for i, item in enumerate(inputs)
            ]

        def add_item(event=None):
            html, widget = create_list_widget(len(inputs))
            if self.steps:
                inputs.append({"widget": widget})
            else:
                inputs.append({"html": html, "widget": widget})
            update_inputs()

        # Initial item
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


    def render_union_input(self, selected_type, field, name, default, provide=True, html_h_tag=2):
        html_h_tag = html_h_tag + 1
        input_widget = self.get_input_widget(selected_type, field, name, default, html_h_tag, provide=provide)
        if not input_widget:
            input_widget = self.render_custom_types(model_class=selected_type, field=field, prefix=name, default_values=default, html_h_tag=html_h_tag)
        return input_widget



    def get_input_widget(self, type_, field, name, default, html_h_tag, provide=True):
        selected_type = type_.lower()
        if self.step_parameters:
            new_names = name.split(".")[1:]
            value = self.step_parameters.copy()
            for new_name in new_names:
                value = value.get(new_name)
            if value is not None:
                default = value
        if not provide:
            return pn.widgets.TextInput(value="", disabled=True)
        if selected_type == "int":
            return pn.widgets.IntInput(name=name, value=default)
        if selected_type == "float":
            return pn.widgets.FloatInput(name=name, value=default)
        if selected_type == "bool":
            return pn.widgets.Checkbox(name=name, value=default)
        if selected_type == "dict":
            default = {"key": "value"} if default is None or default == {} else default
            return pn.widgets.JSONEditor(name=name,value=default, width=400, mode="tree")
        if selected_type == "str":
            return pn.widgets.TextInput(name=name, value=default)
        if selected_type == "Not provided (NoneType)":
            return pn.pane.HTML(
                "<div class='ppy-pn-description ppy-pn-italic'>No input needed</div>"
            )
        if selected_type in ["literal"]:
            opts = field["field_options"].get("literal", []) if "field_options" in field else field.get("options", [])
            return pn.widgets.Select(name=name, options=opts, value=default)
        if selected_type == "list":
            item_type = field.get("item_type", "str")
            item_default = field.get("item_default", "")

            return self.render_list_input(field, name, html_h_tag=html_h_tag) if item_type else pn.widgets.TextInput(name=name, value=item_default)
        return None


    def render_method_with_mode_params(self, field_meta,
                                    model_class,
                                    prefix=None,
                                    html_h_tag=2,
                                    widgets=None):
        mode_options = field_meta["mode"]["options"]

        mode_select = pn.widgets.Select(
            name=f"{prefix}.mode",
            options=mode_options,
            value=mode_options[0],

        )

        def get_param_model(model_class):
            mapping = {
                "SamplingConfig": SAMPLING_LAYER_METHODS,
                "PositionConfig": POSITION_LAYER_METHODS,
                "ColourConfig": COLOUR_LAYER_METHODS,
                "ConvertConfig": CONVERT_LAYER_METHODS,
            }
            return mapping.get(model_class)



        def make_params_widget(selected_mode):
            param_model_name = get_param_model(model_class.__name__)[selected_mode]["params"]
            if param_model_name:
                return self.render_custom_types(model_class=param_model_name.__name__,
                                                prefix=f"{prefix}.params",
                                                html_h_tag=html_h_tag + 1)
            return pn.pane.Markdown(f"*No parameters available for mode `{selected_mode}`*")

        params_widget = pn.bind(make_params_widget, mode_select)
        if widgets is None:
            widgets = []

        widgets.extend([pn.Column(
            pn.pane.HTML(
                f"<div class='ppy-pn-title-{html_h_tag} ppy-pn-bold'>Mode</div>"
                f"<div class='ppy-pn-description'>{field_meta['mode'].get('description', '')}</div>"
            ),
            mode_select,
            pn.pane.HTML(
                f"<div class='ppy-pn-title-{html_h_tag} ppy-pn-bold'>Params</div>"
                f"<div class='ppy-pn-description'>{field_meta['params'].get('description', '')}</div>"
            ),
            params_widget
        )])
        return widgets

    def widget_from_literal(self, field_name: str, literal_type: Literal) -> pn.widgets.Select:
        options = list(get_args(literal_type))
        return pn.widgets.Select(name=field_name, options=options, value=options[0])
