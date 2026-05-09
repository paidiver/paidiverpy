"""Focused branch-coverage tests for configuration, pipeline, metadata parser, and HTML helpers."""

# ruff: noqa

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from IPython.display import HTML
from jsonschema.exceptions import ValidationError
from shapely import Polygon

from paidiverpy.config import configuration as configuration_module
from paidiverpy.config.configuration import Configuration
from paidiverpy.metadata_parser import metadata_parser as metadata_module
from paidiverpy.metadata_parser.metadata_parser import MetadataParser
from paidiverpy.pipeline import pipeline as pipeline_module
from paidiverpy.pipeline.pipeline import Pipeline
from paidiverpy.utils import formating_html
from tests.utils import DummyLogger, DummyStep


def _make_pipeline_stub() -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.logger = DummyLogger()
    pipeline.images = SimpleNamespace(images=[], remove_steps_by_order=lambda *_: None)
    pipeline.metadata = SimpleNamespace(metadata=pd.DataFrame({"flag": [0, 3]}))
    pipeline.steps = []
    pipeline.runned_steps = -1
    pipeline.config = SimpleNamespace(add_step=lambda *_a, **_k: None, general=SimpleNamespace())
    pipeline._get_step_name = lambda _cls: "sampling"
    return pipeline


def test_configuration_validation_add_remove_export_and_repr(monkeypatch, tmp_path: Path):
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    with pytest.raises(ValueError, match="No steps to remove"):
        config.remove_step()

    config.add_step(
        parameters={"name": "sample", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
        validate=False,
    )
    with pytest.raises(ValueError, match="Invalid step index"):
        config.remove_step(99)

    exported = config.export(None)
    assert isinstance(exported, str)
    assert "general:" in exported

    out_file = tmp_path / "cfg.yml"
    assert config.export(out_file) is None
    assert out_file.exists()

    monkeypatch.setattr(formating_html, "config_repr", lambda _cfg: "<div>cfg</div>")
    assert config._repr_html_() == "<div>cfg</div>"


def test_configuration_add_general_validate_rolls_back(monkeypatch):
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    original = config.general.name

    monkeypatch.setattr(
        Configuration,
        "validate_config",
        staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(ValidationError("bad"))),
    )

    with pytest.raises(ValidationError, match="Failed to validate the general config"):
        config.add_general({"name": "changed-name"}, validate=True)

    assert config.general.name == original


def test_configuration_validate_config_remote_schema_and_error(monkeypatch, tmp_path: Path):
    class _FakeValidatorOk:
        def __init__(self, _schema):
            pass

        def iter_errors(self, _config):
            return []

    class _FakeError:
        path = []
        message = "invalid"

        def __str__(self):
            return "fake-error"

    class _FakeValidatorFail:
        def __init__(self, _schema):
            pass

        def iter_errors(self, _config):
            return [_FakeError()]

    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("general: {}\nsteps: []\n", encoding="utf-8")

    monkeypatch.setattr(configuration_module, "path_is_remote", lambda _path: True)
    monkeypatch.setattr(configuration_module, "get_file_from_bucket", lambda *_a, **_k: b"{}")
    monkeypatch.setattr(configuration_module, "Draft202012Validator", _FakeValidatorOk)
    Configuration.validate_config(cfg_file, local=False)

    monkeypatch.setattr(configuration_module, "Draft202012Validator", _FakeValidatorFail)
    with pytest.raises(ValidationError, match="Failed to validate"):
        Configuration.validate_config({"general": {}, "steps": []}, local=False)


def test_configuration_load_steps_invalid_name_and_output_path_docker(monkeypatch):
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    with pytest.raises(ValueError, match="Invalid step name"):
        config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})

    monkeypatch.setattr(configuration_module, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    out, is_remote = config.get_output_path(output_path="local-output")
    assert str(out) == "/app/output"
    assert is_remote is False


def test_pipeline_validate_process_custom_algorithm_and_helpers(monkeypatch):
    pipeline = _make_pipeline_stub()

    with pytest.raises(ValueError, match="No steps defined for the pipeline"):
        pipeline._validate_pipeline()

    install_calls = []

    class _CustomLayer:
        pass

    _CustomLayer.__name__ = "CustomLayer"
    pipeline.steps = [("custom", _CustomLayer, {"dependencies": "a", "dependencies_path": "deps.txt"})]
    monkeypatch.setattr(pipeline_module, "check_and_install_dependencies", lambda deps, dep_path: install_calls.append((deps, dep_path)))
    pipeline._validate_pipeline()
    assert install_calls == [("a", "deps.txt")]

    with pytest.raises(ValueError, match="File path not provided"):
        pipeline.process_custom_algorithm({"name": "algo", "class_name": "Algo"}, 0)

    monkeypatch.setattr(pipeline, "load_custom_algorithm", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("x")))
    with pytest.raises(FileNotFoundError, match="not found for custom algorithm"):
        pipeline.process_custom_algorithm({"name": "algo", "class_name": "Algo", "file_path": "f.py"}, 0)

    monkeypatch.setattr(pipeline, "load_custom_algorithm", lambda *_a, **_k: (_ for _ in ()).throw(AttributeError("x")))
    with pytest.raises(AttributeError, match="Class Algo not found"):
        pipeline.process_custom_algorithm({"name": "algo", "class_name": "Algo", "file_path": "f.py"}, 0)

    assert pipeline._get_steps_params(("raw", object)) == ("raw", object, {})


def test_pipeline_validate_from_step_and_add_step_error_paths(monkeypatch):
    pipeline = _make_pipeline_stub()

    with pytest.raises(ValueError, match="cannot run the pipeline from a specific step"):
        pipeline._validate_from_step(1)

    removed_orders = []
    pipeline.images = SimpleNamespace(images=[1, 2, 3], remove_steps_by_order=lambda order: removed_orders.append(order))
    pipeline.metadata = SimpleNamespace(metadata=pd.DataFrame({"flag": [0, 2, 6]}))
    pipeline._validate_from_step(1)
    assert pipeline.runned_steps == 1
    assert removed_orders == [2]
    assert pipeline.metadata.metadata["flag"].tolist() == [0, 2, 0]

    pipeline._validate_from_step(99)

    with pytest.raises(ValueError, match="To substitute a step you need to provide the index"):
        pipeline.add_step("name", object, {}, substitute=True)

    pipeline.steps = [("s0", object, {}), ("s1", object, {})]
    pipeline.config = SimpleNamespace(add_step=lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad-step")))
    with pytest.raises(ValueError, match="Invalid step parameters"):
        pipeline.add_step("name", object, {}, index=1, substitute=False)

    monkeypatch.setattr(formating_html, "pipeline_repr", lambda _p: "<div>pipe</div>")
    assert pipeline._repr_html_() == "<div>pipe</div>"


def test_formating_html_repr_and_pipeline_branches():
    raw_obj = object()
    rendered = formating_html._obj_repr(raw_obj, "<b>body</b>", html=False)
    assert isinstance(rendered, str)
    assert "ppy-text-repr-fallback" in rendered

    rendered_html = formating_html._obj_repr(raw_obj, "<b>body</b>", html=True)
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
    da = pytest.importorskip("dask.array")
    dask_img = da.from_array(np.ones((2, 2, 1), dtype=np.uint8), chunks=(2, 2, 1))
    b64_grey = formating_html.numpy_array_to_base64(dask_img, size=(1, 1))
    assert b64_grey.startswith("data:image/jpeg;base64,")

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[:, :, 3] = 1
    b64_rgba = formating_html.numpy_array_to_base64(rgba, size=None)
    assert b64_rgba.startswith("data:image/png;base64,")

    json_html = formating_html._json_to_html({"k": [1, "x", None]})
    assert "ppy-json-block" in json_html

    yaml_html = formating_html._yaml_to_html({"k": [{"a": True}, 4]})
    assert "ppy-json-key" in yaml_html

    assert "ppy-json-string" in formating_html.style_json_yaml("str")
    assert "ppy-json-number" in formating_html.style_json_yaml(3)
    assert "ppy-json-number" in formating_html.style_json_yaml(True)
    assert "ppy-json-null" in formating_html.style_json_yaml(None)
    assert "[" in formating_html.style_json_yaml(pd.Series([1, 2]))
    assert "ppy-json-string" in formating_html.style_json_yaml(Polygon([(0, 0), (1, 0), (0, 1)]))
    assert isinstance(formating_html.style_json_yaml(object()), str)


def test_metadata_parser_csv_paths_and_conversion(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "filename,image-datetime,image-latitude,image-longitude\n"
        "img2.jpg,2024-01-02T00:00:00,10.0,20.0\n"
        "img1.jpg,2024-01-01T00:00:00,11.0,21.0\n",
        encoding="utf-8",
    )

    general = SimpleNamespace(
        metadata_path=str(csv_path),
        metadata_type="CSV_FILE",
        append_data_to_metadata=None,
        metadata_conventions=None,
        sample_data=None,
    )
    parser = MetadataParser(config=SimpleNamespace(general=general), use_dask=False)

    assert "ID" in parser.metadata.columns
    assert parser.metadata["filename"].tolist() == ["img1.jpg", "img2.jpg"]
    assert "point" in parser.metadata.columns

    no_dt = parser._handle_datetime(pd.DataFrame({"filename": ["a.jpg"]}))
    assert "filename" in no_dt.columns

    with pytest.raises(ValueError, match="Unsupported output format"):
        parser.export_metadata(output_format="xml", output_path=str(tmp_path / "out"))

    raw_meta = pd.DataFrame(
        {
            "filename": ["a.jpg", "b.jpg"],
            "flag": [0, 2],
            "point": ["x", "y"],
            "value": [1, None],
        }
    )
    MetadataParser.convert_metadata_to({"dataset": "d1"}, raw_meta, str(tmp_path / "meta_out"), "csv", from_step=1)
    out_csv = tmp_path / "meta_out.csv"
    assert out_csv.exists()

    MetadataParser.convert_metadata_to({"dataset": "d1"}, raw_meta, str(tmp_path / "meta_out_json"), "json", from_step=-1)
    out_json = tmp_path / "meta_out_json.json"
    assert out_json.exists()

    with pytest.raises(NotImplementedError, match="Croissant format is not implemented yet"):
        MetadataParser.convert_metadata_to({"dataset": "d1"}, raw_meta, str(tmp_path / "meta_out_c"), "croissant")



def test_metadata_parser_ifdo_remote_and_additional_data_branches(tmp_path: Path, monkeypatch):
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "s3://bucket/meta.json"
    parser.storage_options = {"service_name": "s3"}
    parser._validate_ifdo = lambda _m: None

    ifdo_payload = {
        "image-set-header": {"image-set-name": "set"},
        "image-set-items": {
            "img_a.jpg": {"image-datetime": "2024-01-01T00:00:00"},
            "img_b.jpg": {"image-datetime": "2024-01-02T00:00:00"},
        },
    }
    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: True)
    monkeypatch.setattr(metadata_module, "get_file_from_bucket", lambda *_a, **_k: json.dumps(ifdo_payload).encode("utf-8"))
    remote_df = MetadataParser._open_ifdo_metadata(parser)
    assert "filename" in remote_df.columns
    assert "ID" in remote_df.columns

    parser.append_data_to_metadata = str(tmp_path / "append_bad.csv")
    base = pd.DataFrame({"filename": ["img_a.jpg"]})
    (tmp_path / "append_bad.csv").write_text("col\n1\n", encoding="utf-8")
    parser._rename_columns = lambda *_a, **_k: (_ for _ in ()).throw(ValueError("missing filename"))
    unchanged = MetadataParser._add_data_to_metadata(parser, base)
    assert unchanged.equals(base)

    parser.append_data_to_metadata = str(tmp_path / "append_good.csv")
    (tmp_path / "append_good.csv").write_text("filename,extra\nimg_a.jpg,1\nimg_a.jpg,2\n", encoding="utf-8")
    parser._rename_columns = lambda metadata, *_a, **_k: metadata
    merged = MetadataParser._add_data_to_metadata(parser, base)
    assert merged["extra"].iloc[0] == 1

    monkeypatch.setattr(metadata_module, "validate_ifdo", lambda **_k: [{"path": ["image-set-header"], "message": "bad"}])
    monkeypatch.setattr(metadata_module, "format_ifdo_validation_error", lambda _path: "path")
    parser.logger = DummyLogger()
    MetadataParser._validate_ifdo(parser, {"image-set-header": {}})

    grouped = MetadataParser.group_metadata_and_dataset_metadata(pd.DataFrame({"filename": ["f1"]}), {"dataset": "abc"})
    assert "dataset" in grouped.columns
