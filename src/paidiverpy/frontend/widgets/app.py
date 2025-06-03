"""Paidiverpy App: Interactive Pipeline Builder and Image Processor."""

from collections.abc import Callable
from io import StringIO
from pathlib import Path
import panel as pn
from paidiverpy.config.configuration import Configuration
from paidiverpy.frontend.json_dump import extract_json
from paidiverpy.frontend.parse import parse_fields_from_pydantic_model
from paidiverpy.frontend.render import WidgetRenderer
from paidiverpy.frontend.widgets.config_general import AppGeneral
from paidiverpy.frontend.widgets.utils import create_title
from paidiverpy.pipeline.pipeline import Pipeline
from paidiverpy.pipeline.pipeline_params import STEPS_CLASS_TYPES
from paidiverpy.utils.exceptions import raise_value_error
from paidiverpy.utils.schema_json_handler import ConfigModel


class App:
    """Main application class for the Paidiverpy frontend."""

    def __init__(self):
        self.pipeline = None
        self.run_pipeline_button = pn.widgets.Button(name="Run Pipeline", button_type="success", disabled=True)
        self.yaml_output = ""
        self.layout = None
        self.code_output = ""
        self.expanded = {"general": True, "steps": True, "images": True, "text_outputs": True}
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

    def create_pipeline_widget(self) -> None:
        """Create the pipeline widget to display the current pipeline configuration."""
        if self.pipeline:
            html = self.pipeline._repr_html_()
            self.pipeline_widget.clear()
            self.pipeline_widget.append(self.run_pipeline_button)
            self.pipeline_widget.append(pn.pane.HTML(html, sizing_mode="stretch_width"))
        else:
            self.pipeline_widget = pn.Column(pn.pane.Markdown("### Pipeline not yet created. Please add configuration to the pipeline first."))

    def create_modal(
        self, title: str = "", information: str = "", on_cancel: bool = False, on_confirm: Callable | None = None, visible: bool = False
    ) -> pn.Column:
        """Create a modal dialog for confirmation actions.

        Args:
            title (str): The title of the modal.
            information (str): The information to display in the modal.
            on_cancel (bool): Whether to attach a cancel action.
            on_confirm (callable, optional): A callback function for confirmation action.
            visible (bool): Whether the modal should be visible initially.

        Returns:
            pn.Column: A Panel Column containing the modal dialog.
        """
        title_pane = create_title(title, html_h_tag=2)
        information_pane = create_title(information, html_h_tag=3, bold=False)
        modal_cancel_button = pn.widgets.Button(name="Cancel", button_type="default", width=100, margin=(10, 0, 0, 0))
        modal_confirm_button = pn.widgets.Button(name="Confirm", button_type="danger", width=100, margin=(10, 0, 0, 0))

        if on_cancel:
            modal_cancel_button.on_click(lambda _: self.template.main[2].__setattr__("visible", False))
        if on_confirm:
            modal_confirm_button.on_click(on_confirm)

        return pn.Column(
            title_pane, information_pane, pn.Row(modal_confirm_button, modal_cancel_button), css_classes=["ppy-pn-danger-modal"], visible=visible
        )

    def update_modal(self, title: str, information: str, on_confirm: Callable | None = None, on_cancel: Callable | None = None) -> None:
        """Update the modal dialog with new content and callbacks.

        Args:
            title (str): The new title for the modal.
            information (str): The new information message for the modal.
            on_confirm (callable, optional): A callback function for confirmation action.
            on_cancel (callable, optional): A callback function for cancellation action.
        """
        self.modal.objects[0].object = title
        self.modal.objects[1].object = information

        # Set new callbacks
        confirm_btn = self.modal.objects[2][0]
        cancel_btn = self.modal.objects[2][1]

        if on_confirm:
            confirm_btn.on_click(on_confirm)
        if on_cancel:
            cancel_btn.on_click(on_cancel)

    def update_alert(self, information: str = "", title: str = "", alert_type: str = "success", visible: bool = True) -> None:
        """Update the alert widget with a message.

        Args:
            information (str): The information message to display.
            title (str): The title of the alert.
            alert_type (str): The type of alert (e.g., "success", "danger").
            visible (bool): Whether the alert should be visible.
        """
        text = ""
        if title:
            text += f"<strong>{title}</strong><br>"
        if information:
            text += information
        self.alert.object = text
        self.alert.alert_type = alert_type
        self.alert.visible = visible

        def hide_alert() -> None:
            self.alert.visible = False

        if alert_type == "success":
            pn.state.curdoc.add_timeout_callback(hide_alert, 10001)
        else:
            pn.state.curdoc.add_timeout_callback(hide_alert, 10000)

    def create_pipeline_functionality(self) -> None:
        """Create the functionality for running the pipeline."""

        def run_pipeline(event) -> None:  # noqa: ANN001, ARG001
            if not self.pipeline:
                self.pipeline = Pipeline(config=self.configuration)
            self.pipeline.run()
            self.code_widget[2].value += "pipeline.run()\n"
            self.update_alert("Pipeline executed successfully!")
            self.update_images()

        self.run_pipeline_button.on_click(run_pipeline)

    def update_images(self) -> None:
        """Update the images widget with the processed images from the pipeline."""
        if self.pipeline and hasattr(self.pipeline, "images"):
            self.update_images_widget()

    def update_images_widget(self) -> None:
        """Update the images widget to display the processed images."""
        self.images_widget.clear()
        all_html = self.pipeline.images._repr_html_()
        image_all = pn.pane.HTML(all_html, sizing_mode="stretch_width")

        one_html = self.pipeline.images.show(0)
        image_individual = pn.Column(one_html, sizing_mode="stretch_width")
        image_individual.visible = False

        def export_images(event) -> None:  # noqa: ANN001, ARG001
            try:
                self.pipeline.save_images()
                self.update_alert("Images saved successfully to the output path!")
                self.code_widget[2].value += "\n#this is the command to save images\n"
                self.code_widget[2].value += "pipeline.save_images()\n"
            except Exception as e:  # noqa: BLE001
                self.update_alert(f"Error exporting images: {e}", alert_type="danger")

        def show_image(event) -> None:  # noqa: ANN001, ARG001
            if select_image_vis.value in ["", "ALL"]:
                self.images_widget.objects[0][2][0].visible = True
                self.images_widget.objects[0][2][1].visible = False
            else:
                self.images_widget.objects[0][2][0].visible = False
                try:
                    one_html = pn.Column(self.pipeline.images.show(int(select_image_vis.value)), sizing_mode="stretch_width")
                except ValueError:
                    one_html = pn.Column("Please enter a valid image number.", sizing_mode="stretch_width")
                self.images_widget.objects[0][2][1].clear()
                self.images_widget.objects[0][2][1].append(one_html)
                self.images_widget.objects[0][2][1].visible = True

        export_images_button = pn.widgets.Button(name="Save Images", icon="file", button_type="success")
        export_images_button.on_click(export_images)

        try:
            select_image_vis_options = ["ALL", *list(range(len(self.pipeline.images.images[0])))]
        except Exception:  # noqa: BLE001
            select_image_vis_options = ["ALL"]
        select_image_vis = pn.widgets.Select(name="Select a Image Number", options=select_image_vis_options, value="ALL")

        image_show_button = pn.widgets.Button(name="Show Images", button_type="success")
        image_show_button.on_click(show_image)

        self.images_widget.append(
            pn.Column(
                export_images_button,
                pn.Row(
                    select_image_vis,
                    image_show_button,
                    css_classes=["ppy-pn-align-right"],
                ),
                pn.Column(image_all, image_individual),
                sizing_mode="stretch_width",
            )
        )

    def create_template(self) -> pn.template.BootstrapTemplate:
        """Create the main application template with sidebar and main content.

        Returns:
            pn.template.BootstrapTemplate: The main application template.
        """
        title_str = "Paidiverpy App: Interactive Pipeline Builder and Image Processor"
        title_pane = create_title(title_str, html_h_tag=0)
        information_str = (
            "<div class='ppy-pn-title-2'>"
            "Welcome to the Paidiverpy App — an interactive tool to design, run, and export image processing pipelines"
            " using the Paidiverpy package.\n"
            "Use the sidebar to configure general settings and add processing steps one by one. Once your steps are defined,"
            " you can run the pipeline to preview the output images.\n"
            "You can export the processed images and save your pipeline as a configuration file compatible with the Paidiverpy Python package.\n"
            "The sidebar also provides a list of command-line examples to help you use your pipeline outside this tool.\n\n"
            "For more information, please visit our"
            "<a href='https://paidiverpy.readthedocs.io/en/latest/' target='_blank'> DOCUMENTATION</a>"
            "</div>"
        )

        information_pane = pn.pane.Markdown(information_str)

        hero_section = pn.Column(title_pane, information_pane, sizing_mode="stretch_width")
        sidebar_title = create_title("Config Inputs", html_h_tag=0)
        alert = pn.Column(self.alert)

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
                pn.layout.Divider(),
                self.pipeline_widget,
                self.images_widget,
            ],
        )

    def confirm_general_update(self, widgets: pn.Column) -> None:
        """Confirm the update of the general configuration.

        Args:
            widgets (pn.Column): The widgets containing the general configuration inputs.
        """
        json_str = extract_json(widgets)
        result = self.update_general_configuration(json_str)
        if result:
            yaml_str = self.pipeline.export_config()
            self.yaml_widget[3].value = yaml_str
            self.code_widget[2].value = (
                "from paidiverpy.pipeline.pipeline import Pipeline\n"
                "from paidiverpy.config.configuration import Configuration\n"
                f"configuration = Configuration(add_general={json_str})\n"
                "pipeline = Pipeline(config=configuration)\n"
            )

    def create_general_widget(self) -> AppGeneral:
        """Create the general configuration widget to manage the pipeline's general settings.

        Returns:
            AppGeneral: An instance of the AppGeneral class containing the general configuration widget.
        """
        title = "General Configuration Input"

        title_button = pn.widgets.Button(name=self.get_button_name("general", title), width=300, margin=(0, 0, 10, 0))
        title_button.css_classes = ["ppy-pn-header-button"]

        def toggle(event) -> None:  # noqa: ANN001, ARG001
            self.expanded["general"] = not self.expanded["general"]
            self.general_form.visible = self.expanded["general"]
            title_button.name = self.get_button_name("general", title)

        def on_submit(event) -> None:  # noqa: ANN001, ARG001
            if self.general_widget.config:

                def on_confirm() -> None:
                    self.confirm_general_update(self.general_form)
                    self.modal.visible = False

                def on_cancel() -> None:
                    self.modal.visible = False

                self.update_modal(
                    title="Confirm General Configuration Update",
                    information=("Updating General Configuration will erase the current pipeline configuration. Are you sure?"),
                    on_confirm=on_confirm,
                    on_cancel=on_cancel,
                )
                self.modal.visible = True
            else:
                self.confirm_general_update(self.general_form)

        title_button.on_click(toggle)

        general_widget = AppGeneral()
        general_widget.create_widget()
        submit_button = pn.widgets.Button(name="Create/Update General", button_type="success")
        submit_button.on_click(on_submit)

        self.general_form = pn.Column(pn.Row(submit_button), *general_widget.layout, pn.Row(submit_button))

        general_widget.layout = pn.Column(title_button, self.general_form, sizing_mode="stretch_width", css_classes=["ppy-pn-config-form"])

        return general_widget

    def create_steps_widget(self) -> pn.Column:
        """Create the steps widget to manage the steps in the pipeline.

        Returns:
            pn.Column: A Panel Column containing the steps configuration form.
        """
        title = "Steps Configuration Input"

        title_button = pn.widgets.Button(name=self.get_button_name("steps", title), width=300, margin=(0, 0, 10, 0))
        title_button.css_classes = ["ppy-pn-header-button"]

        def toggle(event) -> None:  # noqa: ANN001, ARG001
            self.expanded["steps"] = not self.expanded["steps"]
            self.steps_form.visible = self.expanded["steps"]
            title_button.name = self.get_button_name("steps", title)

        title_button.on_click(toggle)

        self.steps_form = self.create_steps_form()

        return pn.Column(title_button, self.steps_form, sizing_mode="stretch_width", css_classes=["ppy-pn-config-form"])

    def create_steps_form(self) -> pn.Column:
        """Create the steps form for adding or updating steps in the pipeline.

        Returns:
            pn.Column: A Panel Column containing the steps form.
        """
        idx = 0
        steps_layout = []
        if self.configuration.steps:
            for idx, step in enumerate(self.configuration.steps):
                steps_layout.append(self.create_form(idx, step))
            idx += 1
        steps_layout.append(self.create_form(idx))
        return pn.Column(*steps_layout)

    def create_form(self, step_number: int, step_parameters: dict | None = None) -> pn.Column:
        """Create a form for adding or updating a step in the pipeline.

        Args:
            step_number (int): The step number for the form.
            step_parameters (dict | None): Optional parameters for the step if updating.

        Returns:
            pn.Column: A Panel Column containing the form for the step.
        """
        widget_render = WidgetRenderer(steps=True, step_parameters=step_parameters)
        default_params = {"steps": parse_fields_from_pydantic_model(ConfigModel)["steps"]}
        default_params["steps"]["type"] = "list"
        default_params["steps"]["item_type"] = default_params["steps"]["field_options"]["list"]
        del default_params["steps"]["field_options"]
        widgets = [widget_render.create_widget(name, field) for name, field in default_params.items()]

        def toggle_visibility(event) -> None:  # noqa: ANN001, ARG001
            inputs.visible = toggle_button.value
            if step_parameters:
                toggle_button.name = f"Hide Step {step_number + 1}" if toggle_button.value else f"Step {step_number + 1}"
            else:
                toggle_button.name = "Hide New Step Form" if toggle_button.value else "Show New Steps Form"

        def on_submit(event) -> None:  # noqa: ANN001, ARG001
            json_str = extract_json(self.steps_form.objects[step_number], True)
            if step_parameters:
                self.update_step_configuration(json_str, step_number)
            else:
                self.update_step_configuration(json_str)

        submit_button = pn.widgets.Button(name="Add Step", button_type="success")
        submit_button.on_click(on_submit)
        if step_parameters:
            submit_button.name = "Update Step"

        inputs = pn.Column(pn.Row(submit_button), *widgets, pn.Row(submit_button))

        if step_parameters:
            toggle_button = pn.widgets.Toggle(name=f"Step {step_number + 1}", button_type="primary", value=False)
            inputs.visible = False
        else:
            toggle_button = pn.widgets.Toggle(name="Hide New Step Form", button_type="primary", value=True)

        toggle_button.param.watch(toggle_visibility, "value")

        if self.general_widget.config:
            self.run_pipeline_button.disabled = False
        else:
            self.run_pipeline_button.disabled = True

        return pn.Column(toggle_button, inputs)

    def create_images_widget(self) -> pn.Column:
        """Create the images widget to display the processed images.

        Returns:
            pn.Column: A Panel Column containing the images output editor.
        """
        return pn.Column(pn.pane.Markdown(""))

    def create_code_yaml_widget(self) -> pn.Column:
        """Create the code and YAML output widget to display the generated configuration and code.

        Returns:
            pn.Column: A Panel Column containing the YAML and code output editors.
        """
        self.yaml_widget = self.create_yaml_widget()
        self.code_widget = self.create_code_widget()

        title = "Code and Config Outputs"

        title_button = pn.widgets.Button(name=self.get_button_name("text_outputs", title), width=300, margin=(0, 0, 10, 0))
        title_button.css_classes = ["ppy-pn-header-button"]

        def toggle(event) -> None:  # noqa: ANN001, ARG001
            self.expanded["text_outputs"] = not self.expanded["text_outputs"]
            widgets.visible = self.expanded["text_outputs"]
            title_button.name = self.get_button_name("text_outputs", title)

        title_button.on_click(toggle)

        widgets = pn.Column(self.yaml_widget, self.code_widget)

        return pn.Column(title_button, widgets, sizing_mode="stretch_width", css_classes=["ppy-pn-config-form"])

    def create_yaml_widget(self) -> pn.Column:
        """Create the YAML output widget to display the generated configuration.

        Returns:
            pn.Column: A Panel Column containing the YAML output editor and export button.
        """
        title_str = "Config YAML Output"
        title_pane = create_title(title_str, html_h_tag=1)
        information_str = "This section displays the generated YAML configuration based on your inputs. You can copy this YAML for further use."
        information_pane = create_title(information_str, html_h_tag=3, bold=False)

        self.yaml_output_editor = pn.widgets.CodeEditor(
            value=self.yaml_output, language="yaml", theme="monokai", readonly=True, height=300, sizing_mode="stretch_width"
        )

        def export_yaml() -> StringIO | None:
            if not self.pipeline:
                self.update_alert("No configuration available to export.", alert_type="danger")
                return None
            self.pipeline.export_config("pipeline_config.yaml")
            self.code_widget[2].value += "\n# this is the command to export the configuration to a yaml file\n"
            self.code_widget[2].value += "pipeline.export_config('pipeline_config.yaml')\n"
            self.code_widget[2].value += "\n# with the yaml file, you can run the commands below to run the whole pipeline:\n"
            self.code_widget[2].value += "# pipeline = Pipeline(config_file_path='pipeline_config.yaml')\n"
            self.code_widget[2].value += "# pipeline.run()\n"
            with Path("pipeline_config.yaml").open("r") as file:
                sio = StringIO(file.read())
                sio.seek(0)
                return sio

        export_button = pn.widgets.FileDownload(
            name="Export Config", callback=pn.bind(export_yaml), filename="generated_config.yml", button_type="success"
        )

        return pn.Column(title_pane, information_pane, export_button, self.yaml_output_editor, sizing_mode="stretch_width")

    def create_code_widget(self) -> pn.Column:
        """Create the code output widget to display the generated code.

        Returns:
            pn.Column: A Panel Column containing the code output editor.
        """
        title_str = "Code Output"
        title_pane = create_title(title_str, html_h_tag=1)
        information_str = "This section displays the generated code based on your configuration. You can copy this code for further use."
        information_pane = create_title(information_str, html_h_tag=3, bold=False)

        self.code_output_editor = pn.widgets.CodeEditor(
            value=self.code_output, language="python", theme="monokai", readonly=True, height=300, sizing_mode="stretch_width"
        )
        return pn.Column(title_pane, information_pane, self.code_output_editor, sizing_mode="stretch_width")

    def get_button_name(self, expanded: str, title: str) -> str:
        """Get the button name based on the expansion state.

        Args:
            expanded (str): The key in the `self.expanded` dictionary.
            title (str): The title of the section.

        Returns:
            str: The formatted button name with an arrow indicating expansion state.
        """
        arrow = "▼" if self.expanded[expanded] else "▶"
        return f"{title} {arrow}"

    def update_general_configuration(self, json_str: dict) -> bool:
        """Update the general configuration of the pipeline.

        Args:
            json_str (dict): The JSON string containing the general configuration.

        Returns:
            bool: True if the configuration was updated successfully, False otherwise.
        """
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
            self.images_widget.clear()
        except Exception as e:  # noqa: BLE001
            self.update_alert(f"Error: {e}", alert_type="danger")
            return False
        if self.configuration.general:
            self.run_pipeline_button.disabled = False
        else:
            self.run_pipeline_button.disabled = True
        return True

    def update_step_configuration(self, json_str: dict, idx: int | None = None) -> None:
        """Update the step configuration in the pipeline.

        Args:
            json_str (dict): The JSON string containing the step configuration.
            idx (int | None): The index of the step to update. If None, a new step is added.
        """
        try:
            step_layer = next(iter(json_str.keys()))
            name = json_str[step_layer].get("name", f"{step_layer}_{idx}")
            class_name = STEPS_CLASS_TYPES[step_layer]
            if not self.pipeline:
                msg = "Pipeline not initialized. Please create a general config first."
                raise_value_error(msg)
            if idx is not None:
                self.pipeline.add_step(
                    name,
                    class_name,
                    json_str[step_layer],
                    idx + 1,
                    substitute=True,
                )
            else:
                self.pipeline.add_step(name, class_name, json_str[step_layer])

            self.update_alert(f"Step {name} Added Successfully!")
            self.create_pipeline_widget()
            yaml_str = self.pipeline.export_config()
            self.yaml_widget[3].value = yaml_str
            self.code_widget[
                2
            ].value += f"pipeline.add_step(\n    name='{name}',\n    step_class={class_name},\n    parameters={json_str[step_layer]},\n"
            if idx is not None:
                self.code_widget[2].value += f"    index={idx},\n"
                self.code_widget[2].value += "    substitute=True\n"
            self.code_widget[2].value += ")\n"
            self.steps_form.clear()
            updated_form = self.create_steps_form()
            for obj in updated_form:
                self.steps_form.append(obj)
        except Exception as e:  # noqa: BLE001
            self.update_alert(f"Error: {e}", alert_type="danger")

    def show(self) -> None:
        """Display the application."""
        self.template.servable()


if __name__ == "__main__":
    app = App()
    app.show()
