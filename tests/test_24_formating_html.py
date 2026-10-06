"""Unit tests for HTML formatting utilities in paidiverpy."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from IPython.display import HTML
from shapely import Polygon
from paidiverpy.utils import formating_html
from tests.utils import DummyStep


def test_formating_html_repr_and_pipeline_branches():
    """Test the HTML representation of objects and pipeline branches."""
    raw_obj = object()
    rendered = formating_html._obj_repr(raw_obj, "<b>body</b>", html=False)  # noqa: SLF001
    assert isinstance(rendered, str)
    assert "ppy-text-repr-fallback" in rendered

    rendered_html = formating_html._obj_repr(raw_obj, "<b>body</b>", html=True)  # noqa: SLF001
    assert isinstance(rendered_html, HTML)

    general = SimpleNamespace(name="raw", step_name="open", to_dict=lambda: {"name": "raw", "step_name": "open"})
    steps = [DummyStep("sampling_1", "sampling"), DummyStep("convert_1", "convert")]

    pipeline_one = SimpleNamespace(config=SimpleNamespace(general=general, steps=[]), steps=[("raw", object, {})])
    html_one = formating_html.pipeline_repr(pipeline_one)
    assert "ppy-pipeline-wrap" in html_one

    pipeline_many = SimpleNamespace(config=SimpleNamespace(general=general, steps=steps), steps=[("raw", object, {}), ("sampling", object, {})])
    html_many = formating_html.pipeline_repr(pipeline_many)
    assert "arrow-right" in html_many

    cfg = SimpleNamespace(to_dict=lambda: {"general": {"name": "raw"}, "steps": []})
    assert "ppy-json-block" in formating_html.config_repr(cfg)


def test_formating_html_image_json_yaml_and_style_helpers():
    """Test the HTML formatting of images, JSON/YAML, and style helpers."""
    da = pytest.importorskip("dask.array")
    dask_img = da.from_array(np.ones((2, 2, 1), dtype=np.uint8), chunks=(2, 2, 1))
    b64_grey = formating_html.numpy_array_to_base64(dask_img, size=(1, 1))
    assert b64_grey.startswith("data:image/jpeg;base64,")

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[:, :, 3] = 1
    b64_rgba = formating_html.numpy_array_to_base64(rgba, size=None)
    assert b64_rgba.startswith("data:image/png;base64,")

    json_html = formating_html._json_to_html({"k": [1, "x", None]})  # noqa: SLF001
    assert "ppy-json-block" in json_html

    yaml_html = formating_html._yaml_to_html({"k": [{"a": True}, 4]})  # noqa: SLF001
    assert "ppy-json-key" in yaml_html

    assert "ppy-json-string" in formating_html.style_json_yaml("str")
    assert "ppy-json-number" in formating_html.style_json_yaml(3)
    assert "ppy-json-number" in formating_html.style_json_yaml(True)
    assert "ppy-json-null" in formating_html.style_json_yaml(None)
    assert "[" in formating_html.style_json_yaml(pd.Series([1, 2]))
    assert "ppy-json-string" in formating_html.style_json_yaml(Polygon([(0, 0), (1, 0), (0, 1)]))
    assert isinstance(formating_html.style_json_yaml(object()), str)
