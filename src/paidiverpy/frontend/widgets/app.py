from io import StringIO
from pathlib import Path
import panel as pn
from jsonschema import ValidationError
from paidiverpy.config.configuration import Configuration
from paidiverpy.frontend.json_dump import extract_json
from paidiverpy.frontend.parse import parse_fields_from_pydantic_model
from paidiverpy.frontend.render import WidgetRenderer
from paidiverpy.frontend.widgets.config_general import AppGeneral
from paidiverpy.frontend.widgets.utils import create_title
from paidiverpy.pipeline.pipeline import Pipeline
from paidiverpy.pipeline.pipeline_params import STEPS_CLASS_TYPES
from paidiverpy.utils.schema_json_handler import ConfigModel


class App:
    def __init__(self):
        self.pipeline = None
        self.run_pipeline_button = pn.widgets.Button(name="Run Pipeline", button_type="primary", disabled=True)
        self.yaml_output = ""
        self.layout = None
        self.code_output = ""
        self.expanded = {
            "general": True,
            "steps": True,
            "images": True,
            "text_outputs": True
        }
        self.configuration = Configuration()

        self.create_pipeline_functionality()
        self.general_widget = self.create_general_widget()
        self.steps_widget = self.create_steps_widget()
        self.images_widget = self.create_images_widget()
        self.pipeline_widget = None
        self.create_pipeline_widget()
        self.code_yaml_widget = self.create_code_yaml_widget()
        self.modal = self.create_modal()
        self.alert = pn.pane.Alert("", alert_type="success", visible=False)

        self.template = self.create_template()

    def create_pipeline_widget(self):

        if self.pipeline:
            html = self.pipeline._repr_html_(only_html=True)
            self.pipeline_widget.clear()
            self.pipeline_widget.append(self.run_pipeline_button)
            self.pipeline_widget.append(pn.pane.HTML(html, sizing_mode="stretch_width"))
        else:
            self.pipeline_widget = pn.Column(pn.pane.Markdown("### Pipeline not yet created. Please add configuration to the pipeline first."))



    def create_modal(self, title="", information="", on_cancel=False, on_confirm=None, visible=False):
        title_pane = create_title(title, html_h_tag=2)
        information_pane = create_title(
            information,
            html_h_tag=3,
            bold=False
        )
        modal_cancel_button = pn.widgets.Button(name="Cancel", button_type="default", width=100, margin=(10, 0, 0, 0))
        modal_confirm_button = pn.widgets.Button(name="Confirm", button_type="danger", width=100, margin=(10, 0, 0, 0))

        # Attach default callbacks if provided
        if on_cancel:
            modal_cancel_button.on_click(lambda event: self.template.main[2].__setattr__("visible", False))
        if on_confirm:
            modal_confirm_button.on_click(on_confirm)

        return pn.Column(
            title_pane,
            information_pane,
            pn.Row(modal_confirm_button, modal_cancel_button),
            css_classes=["ppy-pn-danger-modal"],
            visible=visible
        )

    def update_modal(self, title, information, on_confirm=None, on_cancel=None):
        title_pane = create_title(title, html_h_tag=2)
        information_pane = create_title(
            information,
            html_h_tag=3,
            bold=False
        )
        self.modal.objects[0].object = title  # title_pane
        self.modal.objects[1].object = information   # information_pane

        # Set new callbacks
        confirm_btn = self.modal.objects[2][0]
        cancel_btn = self.modal.objects[2][1]

        if on_confirm:
            confirm_btn.on_click(on_confirm)
        if on_cancel:
            cancel_btn.on_click(on_cancel)


    def update_alert(self, information="", title="", alert_type="success", visible=True):
        text = ""
        if title:
            text += f"<strong>{title}</strong><br>"
        if information:
            text += information
        self.alert.object = text
        self.alert.alert_type = alert_type
        self.alert.visible = visible

        def hide_alert():
            self.alert.visible = False

        pn.state.curdoc.add_timeout_callback(hide_alert, 5000)



    def create_pipeline_functionality(self):
        def run_pipeline(event):
            if not self.pipeline:
                self.pipeline = Pipeline(config=self.configuration)
            self.pipeline.run()
            self.code_widget[2].value +="pipeline.run()\n"
            self.update_alert("Pipeline executed successfully!")
            self.update_images()

        self.run_pipeline_button.on_click(run_pipeline)

    def update_images(self):
        if self.pipeline and hasattr(self.pipeline, "images"):
            self.update_images_widget()

    def update_images_widget(self):
        self.images_widget.clear()
        all_html = self.pipeline.images._repr_html_()
        image_all = pn.pane.HTML(all_html, sizing_mode="stretch_width")

        one_html = self.pipeline.images.show(0)
        image_individual = one_html
        image_individual.visible = False

        def export_images(event):
            try:
                self.pipeline.save_images()
                self.update_alert("Images exported successfully to the output path!")
                self.code_widget[2].value += "pipeline.save_images()\n"
            except Exception as e:
                self.update_alert(f"Error exporting images: {e}", alert_type="danger")

        def show_image(event):
            if select_image_vis.value in ["", "all"]:
                self.images_widget.objects[0][2][0].visible = True
                self.images_widget.objects[0][2][1].visible = False
            else:
                self.images_widget.objects[0][2][0].visible = False
                try:
                    one_html = self.pipeline.images.show(int(select_image_vis.value))
                except ValueError:
                    one_html = pn.pane.HTML("Please enter a valid image number.", sizing_mode="stretch_width")
                self.images_widget.objects[0][2][1].object = one_html
                image_individual.visible = True


        export_images_button = pn.widgets.Button(name="Export Images", icon="file", button_type="primary")
        export_images_button.on_click(export_images)

        select_image_vis = pn.widgets.TextInput(
            name="Select a Image Number",
            value="all",
            placeholder="Enter 'all' to see all images or a specific number"
        )
        image_show_button = pn.widgets.Button(name="Show Image", button_type="primary")
        image_show_button.on_click(show_image)


        self.images_widget.append(pn.Column(
            export_images_button,
            pn.Row(select_image_vis, image_show_button),
            pn.Column(
                image_all,
                image_individual
            ),
            sizing_mode="stretch_width"
        ))


    def create_template(self):
        title_str = "Interactive Pipeline Generator"
        title_pane = create_title(title_str, html_h_tag=0)
        information_str = (
            "This tool allows you to configure the pipeline steps interactively. "
            "You can adjust parameters for each step and generate a YAML configuration file."
        )
        information_pane = create_title(information_str, html_h_tag=3, bold=False)

        # Hero section at the top of the main area
        hero_section = pn.Column(
            title_pane,
            information_pane,
            sizing_mode="stretch_width"
        )
        # Create a title for the sidebar
        sidebar_title = create_title("Config Info", html_h_tag=0)
        alert = pn.Column(self.alert)

        # Use MaterialTemplate
        return pn.template.BootstrapTemplate(
            title="Paidiverpy App",
            sidebar=[
                sidebar_title,
                self.general_widget.layout,
                self.steps_widget,
                pn.Row(self.code_yaml_widget),
            ],
            main=[
                hero_section,
                pn.layout.Divider(),
                alert,
                self.modal,
                self.pipeline_widget,
                self.images_widget,
            ]
        )

    def confirm_general_update(self, widgets, modal=False):
        json_str = extract_json(widgets)
        result = self.update_general_configuration(json_str)
        if result:
            self.template.main[2].visible = False
            yaml_str = self.pipeline.export_config()
            self.yaml_widget[3].value = yaml_str
            self.code_widget[2].value = (
                "from paidiverpy.pipeline.pipeline import Pipeline\n"
                "from paidiverpy.config.configuration import Configuration\n"
                f"configuration = Configuration(add_general={json_str})\n"
                "pipeline = Pipeline(config=configuration)\n"
            )

    def create_general_widget(self):

        title = "General Configuration Input"

        title_button = pn.widgets.Button(name=self.get_button_name("general", title), width=300, margin=(0,0,10,0))
        title_button.css_classes = ["ppy-pn-header-button"]


        def toggle(event):
            self.expanded["general"] = not self.expanded["general"]
            self.general_form.visible = self.expanded["general"]
            title_button.name = self.get_button_name("general", title)

        def on_submit(event):

            if self.general_widget.config:
                def on_confirm(event):
                    self.confirm_general_update(self.general_form, True)
                    self.modal.visible = False

                def on_cancel(event):
                    self.modal.visible = False

                self.update_modal(
                    title="Confirm General Configuration Update",
                    information=(
                        "Updating General Configuration will erase the current pipeline "
                        "configuration. Are you sure?"
                    ),
                    on_confirm=on_confirm,
                    on_cancel=on_cancel
                )
                self.modal.visible = True
            else:
                self.confirm_general_update(self.general_form)

        # Toggle visibility function
        title_button.on_click(toggle)

        general_widget = AppGeneral()
        general_widget.create_widget()
        submit_button = pn.widgets.Button(name="Create/Update General", button_type="primary")
        submit_button.on_click(on_submit)

        self.general_form = pn.Column(pn.Row(submit_button), *general_widget.layout, pn.Row(submit_button))

        general_widget.layout = pn.Column(
            title_button,
            self.general_form,
            sizing_mode="stretch_width",
            css_classes=["ppy-pn-config-form"]
        )

        return general_widget

    def create_steps_widget(self):


        title = "Steps Configuration Input"

        title_button = pn.widgets.Button(name=self.get_button_name("steps", title), width=300, margin=(0,0,10,0))
        title_button.css_classes = ["ppy-pn-header-button"]

        def toggle(event):
            self.expanded["steps"] = not self.expanded["steps"]
            self.steps_form.visible = self.expanded["steps"]
            title_button.name = self.get_button_name("steps", title)

        # Toggle visibility function
        title_button.on_click(toggle)

        self.steps_form = self.create_steps_form()

        layout = pn.Column(title_button, self.steps_form, sizing_mode="stretch_width", css_classes=["ppy-pn-config-form"])


        return layout

    def create_steps_form(self):
        idx = 0
        steps_layout = []
        if self.configuration.steps:
            for idx, step in enumerate(self.configuration.steps):
                steps_layout.append(self.create_form(idx, step))
        steps_layout.append(self.create_form(idx))
        return pn.Column(*steps_layout)

    def create_form(self, step_number, step_parameters=None):
        widget_render = WidgetRenderer(steps=True, step_parameters=step_parameters)
        default_params = {"steps": parse_fields_from_pydantic_model(ConfigModel)["steps"]}
        widgets = [widget_render.create_widget(
            name,
            field) for name, field in default_params.items()]

        inputs = pn.Column(*widgets)

        def toggle_visibility(event):
            inputs.visible = toggle_button.value
            if step_parameters:
                toggle_button.name = f"Hide Update Step {step_number} Form" if toggle_button.value else f"Show Update Step {step_number} Form"
            else:
                toggle_button.name = "Hide New Step Form" if toggle_button.value else "Show New Steps Form"

        def on_submit(event):
            json_str = extract_json(self.steps_form, True)
            print(step_parameters)
            print(step_number)
            if step_parameters:
                self.update_step_configuration(json_str, step_number)
            else:
                self.update_step_configuration(json_str)

        if step_parameters:
            toggle_button = pn.widgets.Toggle(name=f"Hide Update Step {step_number} Form", button_type="primary", value=True)
        else:
            toggle_button = pn.widgets.Toggle(name="Hide New Step Form", button_type="primary", value=True)

        toggle_button.param.watch(toggle_visibility, "value")

        submit_button = pn.widgets.Button(name="Add Step", button_type="primary")
        submit_button.on_click(on_submit)

        if self.general_widget.config:
            self.run_pipeline_button.disabled = False
        else:
            self.run_pipeline_button.disabled = True


        if step_parameters:
            submit_button.name = "Update Step"

        return pn.Column(toggle_button, inputs, pn.Row(submit_button))



    def create_images_widget(self):
        return pn.Column(pn.pane.Markdown(""))

    def create_code_yaml_widget(self):

        self.yaml_widget = self.create_yaml_widget()
        self.code_widget = self.create_code_widget()

        title = "Code and Config Outputs"

        title_button = pn.widgets.Button(name=self.get_button_name("text_outputs", title), width=300, margin=(0,0,10,0))
        title_button.css_classes = ["ppy-pn-header-button"]

        def toggle(event):
            self.expanded["text_outputs"] = not self.expanded["text_outputs"]
            widgets.visible = self.expanded["text_outputs"]
            title_button.name = self.get_button_name("text_outputs", title)


        # Toggle visibility function
        title_button.on_click(toggle)

        widgets = pn.Column(self.yaml_widget, self.code_widget)

        return pn.Column(
            title_button,
            widgets,
            sizing_mode="stretch_width",
            css_classes=["ppy-pn-config-form"]
        )

    def create_yaml_widget(self):
        title_str = "Config YAML Output"
        title_pane = create_title(title_str, html_h_tag=1)
        information_str = (
            "This section displays the generated YAML configuration based on your inputs. "
            "You can copy this YAML for further use."
        )
        information_pane = create_title(information_str, html_h_tag=3, bold=False)

        # YAML code editor (read-only)
        self.yaml_output_editor = pn.widgets.CodeEditor(
            value=self.yaml_output,
            language="yaml",
            theme="monokai",
            readonly=True,
            height=300,
            sizing_mode="stretch_width"
        )

        def export_yaml():
            if not self.pipeline:
                self.update_alert("No configuration available to export.", alert_type="danger")
                return None
            self.pipeline.export_config("pipeline_config.yaml")
            # load the file again and save it as sio
            with Path("pipeline_config.yaml").open("r") as file:
                sio = StringIO(file.read())
                sio.seek(0)
                return sio

        export_button = pn.widgets.FileDownload(
            name="Export Config",
            callback=pn.bind(export_yaml), filename="generated_config.yml",
            button_type="success"
        )

        return pn.Column(
            title_pane,
            information_pane,
            export_button,
            self.yaml_output_editor,
            sizing_mode="stretch_width"
        )

    def create_code_widget(self):
        title_str = "Code Output"
        title_pane = create_title(title_str, html_h_tag=1)
        information_str = (
            "This section displays the generated code based on your configuration. "
            "You can copy this code for further use."
        )
        information_pane = create_title(information_str, html_h_tag=3, bold=False)

        # Code editor (read-only)
        self.code_output_editor = pn.widgets.CodeEditor(
            value=self.code_output,
            language="python",
            theme="monokai",
            readonly=True,
            height=300,
            sizing_mode="stretch_width"
        )
        return pn.Column(
            title_pane,
            information_pane,
            self.code_output_editor,
            sizing_mode="stretch_width"
        )

    def get_button_name(self, expanded, title):
        arrow = "▼" if self.expanded[expanded] else "▶"
        return f"{title} {arrow}"

    def update_general_configuration(self, json_str):
        try:
            if self.general_widget.config:
                self.configuration = Configuration(add_general=json_str)
            else:
                self.configuration.add_general(json_str)
            self.pipeline = Pipeline(config=self.configuration)
            self.update_alert("Configuration General Created Successfully!")
            self.expanded["general"] = not self.expanded["general"]
            self.general_form.visible = self.expanded["general"]
            self.expanded["steps"] = not self.expanded["steps"]
            self.steps_form.visible = self.expanded["steps"]
            self.create_pipeline_widget()
            self.general_widget.config = json_str
        except ValidationError as e:
            self.update_alert(f"Validation error: {e}", alert_type="danger")
            return False
        if self.configuration.general:
            self.run_pipeline_button.disabled = False
        else:
            self.run_pipeline_button.disabled = True
        return True

    def update_step_configuration(self, json_str, idx=None):
        try:
            step_layer = next(iter(json_str.keys()))
            name = json_str[step_layer].get("name", f"{step_layer}_{idx}")
            class_name = STEPS_CLASS_TYPES[step_layer]
            if idx is not None:
                self.pipeline.add_step(
                    name,
                    class_name,
                    json_str[step_layer],
                    idx,
                    substitute=True,
                )
            else:
                self.pipeline.add_step(
                    name,
                    class_name,
                    json_str[step_layer]
                )

            self.update_alert(f"Step {name} Added Successfully!")
            self.create_pipeline_widget()
            yaml_str = self.pipeline.export_config()
            self.yaml_widget[3].value = yaml_str
            self.code_widget[2].value += (
                f"pipeline.add_step(\n"
                f"    name='{name}',\n"
                f"    step_class={class_name},\n"
                f"    parameters={json_str[step_layer]},\n"
            )
            if idx is not None:
                self.code_widget[2].value += f"    index={idx},\n"
                self.code_widget[2].value += "    substitute=True\n"
            self.code_widget[2].value += ")\n"
            self.steps_form.clear()
            updated_form = self.create_steps_form()
            for obj in updated_form:
                self.steps_form.append(obj)
            # self.steps_widget.objects[1] = self.steps_form
        except ValidationError as e:
            self.update_alert(f"Validation error: {e}", alert_type="danger")


    def show(self):
        self.template.servable()

if __name__ == "__main__":
    app = App()
    app.show()
