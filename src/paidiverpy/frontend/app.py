"""Paidiverpy Panel Application Entry Point."""

from importlib.resources import files
import panel as pn
from paidiverpy.frontend.widgets.app import App
from paidiverpy.utils.formating_html import EXTERNAL_CSS
from paidiverpy.utils.formating_html import EXTERNAL_JS

css_text = files("paidiverpy.static.css").joinpath("style.css").read_text()
js_text = files("paidiverpy.static.js").joinpath("script.js").read_text()

raw_css = [css_text]
css_files = list(EXTERNAL_CSS)
js_files = {
    "external": EXTERNAL_JS[0],
    "inline": "https://raw.githubusercontent.com/paidiver/paidiverpy/refs/heads/186-enhance-exif-handling-in-processed-images/src/paidiverpy/static/js/script.js",
}
pn.extension("jsoneditor", "codeeditor", raw_css=raw_css, css_files=css_files, js_files=js_files)

app = App()
app.show()
