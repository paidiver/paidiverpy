"""Paidiverpy Panel Application Entry Point."""
from importlib.resources import files
import panel as pn
from paidiverpy.frontend.widgets.app import App
from paidiverpy.utils.formating_html import EXTERNAL_CSS
from paidiverpy.utils.formating_html import EXTERNAL_JS

pn.extension("jsoneditor", "codeeditor")
css_text = files("paidiverpy.static.css").joinpath("style.css").read_text()
js_text = files("paidiverpy.static.js").joinpath("script.js").read_text()

pn.config.raw_css.append(css_text)
pn.config.css_files.append(EXTERNAL_CSS)
# pn.config.js_files=[
#     f"<script src='{EXTERNAL_JS[0]}'></script>",
#     f"<script>{EXTERNAL_JS[1]};</script>"
# ]

pn.config.js_files={
    "external": EXTERNAL_JS[0],
    "inline": files("paidiverpy.static.js").joinpath("script.js")
}


app = App()
app.show()
