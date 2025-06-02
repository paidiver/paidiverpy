import panel as pn
from paidiverpy.config.general_config import GeneralConfig
from paidiverpy.frontend.parse import parse_default_params
from paidiverpy.frontend.render import WidgetRenderer
from paidiverpy.frontend.widgets.utils import create_title


class AppGeneral:
    def __init__(self):
        self.has_general = False
        self.has_steps = False
        self.config = None
        self.layout = None
        self.default_params = parse_default_params(GeneralConfig)

    def create_widget(self):
        widget_render = WidgetRenderer()
        widgets = [widget_render.create_widget(name, field) for name, field in self.default_params.items()]

        # information_str = (
        #     "IMPORTANT: The fields 'input_path' and 'metadata_path' are required if 'sample_data' is not provided"
        # )
        # information_pane = create_title(information_str, html_h_tag=3)

        # self.layout = pn.Column(information_pane, *widgets, visible=True)
        self.layout = pn.Column(*widgets, visible=True)
