"""Module for creating the general configuration widget in the PaidiverPy frontend."""

import panel as pn
from paidiverpy.frontend.parse import parse_default_params
from paidiverpy.frontend.render import WidgetRenderer
from paidiverpy.models.general_config import GeneralConfig


class AppGeneral:
    """Class for creating the general configuration widget in the PaidiverPy frontend."""

    def __init__(self):
        self.has_general = False
        self.has_steps = False
        self.config = None
        self.layout = None
        self.default_params = parse_default_params(GeneralConfig)

    def create_widget(self) -> None:
        """Create the general configuration widget layout."""
        widget_render = WidgetRenderer()
        widgets = [widget_render.create_widget(name, field) for name, field in self.default_params.items()]
        self.layout = pn.Column(*widgets, visible=True)
