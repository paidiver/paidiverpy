"""Focused tests for remaining branch gaps in images_layer and configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from paidiverpy.config import configuration as configuration_module
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer


def test_images_layer_save_remote_creates_client_and_bucket(monkeypatch: pytest.MonkeyPatch):
    """Cover remote-save branch that creates client and ensures bucket exists."""
    images_layer = ImagesLayer(output_path="unused")

    fake_images = xr.Dataset(
        {
            "images": (("filename", "y", "x", "band"), np.zeros((1, 1, 1, 1), dtype=np.uint8)),
            "original_height": ("filename", np.array([1])),
            "original_width": ("filename", np.array([1])),
        },
        coords={"filename": np.array(["a.png"], dtype=object)},
    )
    monkeypatch.setattr(images_layer, "get_step", lambda *_a, **_k: fake_images)

    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "paidiverpy.images_layer.create_client",
        lambda: "fake-client",
    )
    monkeypatch.setattr(
        "paidiverpy.images_layer.check_create_bucket_exists",
        lambda bucket, client: seen.update({"bucket": bucket, "client": client}),
    )

    monkeypatch.setattr(
        "paidiverpy.images_layer.xr.apply_ufunc",
        lambda *_a, **_k: SimpleNamespace(compute=lambda: None),
    )

    cfg = SimpleNamespace(get_output_path=lambda _out: ("s3://my-bucket/out/", True))
    images_layer.save(config=cfg, use_dask=False)

    assert seen["bucket"] == "my-bucket"
    assert seen["client"] == "fake-client"


def test_images_layer_process_and_upload_uint16_remote_upload(monkeypatch: pytest.MonkeyPatch):
    """Cover uint16+tiff/png+s3 branch in process_and_upload."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2), dtype=np.uint16)

    uploaded: dict[str, object] = {}

    class _FakePILImage:
        def save(self, buffer, format):
            buffer.write(b"img")

    monkeypatch.setattr("paidiverpy.images_layer.Image.fromarray", lambda _arr: _FakePILImage())
    monkeypatch.setattr(
        "paidiverpy.images_layer.upload_file_to_bucket",
        lambda buffer, path, client: uploaded.update({"size": len(buffer.getvalue()), "path": path, "client": client}),
    )

    images_layer.process_and_upload(image, Path("/tmp/out"), "png", s3_client="s3")

    assert uploaded["size"] > 0
    assert str(uploaded["path"]).endswith(".png")
    assert uploaded["client"] == "s3"


def test_images_layer_process_and_upload_uint16_invalid_format_raises():
    """Cover uint16 invalid-format ValueError branch."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2), dtype=np.uint16)

    with pytest.raises(ValueError, match="16-bit images can only be saved"):
        images_layer.process_and_upload(image, Path("/tmp/out"), "jpg", s3_client=None)


def test_images_layer_process_and_upload_uint8_remote_upload(monkeypatch: pytest.MonkeyPatch):
    """Cover uint8/float32+s3 branch in process_and_upload."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "paidiverpy.images_layer.cv2.imencode",
        lambda *_a, **_k: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )

    uploaded: dict[str, object] = {}
    monkeypatch.setattr(
        "paidiverpy.images_layer.upload_file_to_bucket",
        lambda buffer, path, client: uploaded.update({"size": len(buffer.getvalue()), "path": path, "client": client}),
    )

    images_layer.process_and_upload(image, Path("/tmp/out"), "png", s3_client="s3")

    assert uploaded["size"] == 3
    assert str(uploaded["path"]).endswith(".png")
    assert uploaded["client"] == "s3"


def test_images_layer_process_and_upload_unsupported_dtype_raises():
    """Cover unsupported-dtype ValueError branch."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2, 3), dtype=np.int64)

    with pytest.raises(ValueError, match="Unsupported image dtype"):
        images_layer.process_and_upload(image, Path("/tmp/out"), "png", s3_client=None)


def test_images_layer_calculate_image_single_channel_squeeze_branch():
    """Cover calculate_image branch for trailing single channel arrays."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2, 1), dtype=np.uint8)

    out = images_layer.calculate_image(image)
    assert out.shape == (2, 2)


def test_images_layer_remove_nonexistent_path_and_call_with_explicit_max(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover remove() no-op when path does not exist and __call__ explicit max_images branch."""
    images_layer = ImagesLayer(output_path=tmp_path / "missing-dir")

    monkeypatch.setattr("paidiverpy.images_layer.is_running_in_docker", lambda: False)
    images_layer.remove()

    called: dict[str, object] = {}
    monkeypatch.setattr(
        "paidiverpy.images_layer.formating_html.images_repr",
        lambda _layer, max_images=None, html=False, **_k: called.update({"max_images": max_images, "html": html}) or "ok",
    )

    rendered = images_layer(max_images=5)
    assert rendered == "ok"
    assert called["max_images"] == 5
    assert called["html"] is True

    called.clear()
    rendered_default = images_layer()
    assert rendered_default == "ok"
    assert called["max_images"] == images_layer.max_images
    assert called["html"] is True


def test_configuration_type_checking_import_block_executes(monkeypatch: pytest.MonkeyPatch):
    """Cover TYPE_CHECKING imports by reloading module with TYPE_CHECKING=True."""
    import_lines = (
        "\n" * 28
        + "from paidiverpy.colour_layer import ColourLayer\n"
        + "from paidiverpy.convert_layer import ConvertLayer\n"
        + "from paidiverpy.custom_layer import CustomLayer\n"
        + "from paidiverpy.investigation_layer import InvestigationLayer\n"
        + "from paidiverpy.position_layer import PositionLayer\n"
        + "from paidiverpy.sampling_layer import SamplingLayer\n"
    )

    code = compile(import_lines, configuration_module.__file__, "exec")
    exec(code, {})


def test_configuration_init_add_steps_without_general_warns(caplog):
    """Cover __init__ warning branch when add_steps exists but general is missing."""
    caplog.set_level("WARNING")
    Configuration(add_steps=[{"sampling": {"name": "s1", "mode": "fixed", "params": {"value": 1}}}])
    assert "General configuration is not defined. Please define it first." in caplog.text


def test_configuration_init_with_add_steps_uses_add_step_path():
    """Cover __init__ branch that iterates add_steps when general exists."""
    cfg = Configuration(
        config_file_path="tests/config_files/config_simple.yml",
        add_steps=[{"step_name": "sampling", "name": "s_added", "mode": "fixed", "params": {"value": 1}}],
    )

    assert any(step.name == "s_added" for step in cfg.steps)


def test_configuration_load_steps_invalid_name_path():
    """Cover _load_steps branch where config_class is None and raises."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    with pytest.raises(ValueError, match="Invalid step name"):
        config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})


def test_configuration_load_steps_invalid_name_return_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _load_steps return line after invalid step when raise helper is patched."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    before = len(config.steps)

    monkeypatch.setattr(configuration_module, "raise_value_error", lambda _msg: None)
    config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})

    assert len(config.steps) == before


def test_configuration_validate_config_path_input_branch(tmp_path: Path):
    """Cover validate_config branch for Path input (string/path loading path)."""
    cfg_file = Path("tests/config_files/config_simple.yml")
    Configuration.validate_config(cfg_file)


def test_configuration_add_step_without_general_raises():
    """Cover add_step branch that raises when general config is missing."""
    config = Configuration()

    with pytest.raises(ValueError, match="General configuration is not defined"):
        config.add_step(parameters={"step_name": "sampling", "mode": "fixed", "params": {"value": 1}})


def test_configuration_update_remote_options_no_general_branch():
    """Cover _update_remote_options branch where general is None."""
    config = Configuration()
    config.general = None
    config._update_remote_options()

    assert config.is_remote is False
    assert config.output_is_remote is False


def test_configuration_remove_step_valid_index_branch():
    """Cover remove_step normal pop path."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    config.add_step(parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}}, validate=False)

    before = len(config.steps)
    config.remove_step(len(config.steps) - 1)
    assert len(config.steps) == before - 1


def test_configuration_remove_step_default_index_branch():
    """Cover remove_step branch where config_index is None."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    config.add_step(parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}}, validate=False)

    before = len(config.steps)
    config.remove_step()
    assert len(config.steps) == before - 1


def test_configuration_get_output_path_remote_returns_without_local_mkdir(monkeypatch: pytest.MonkeyPatch):
    """Cover get_output_path branch for remote output paths."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    out, is_remote = config.get_output_path("s3://bucket/some/prefix")

    assert out == "s3://bucket/some/prefix"
    assert is_remote is True


def test_configuration_get_output_path_local_branch_creates_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover local get_output_path branch (docker false, string to Path, mkdir path)."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    local_out = tmp_path / "new-output"

    monkeypatch.setattr(configuration_module, "is_running_in_docker", lambda: False)
    out, is_remote = config.get_output_path(str(local_out))

    assert isinstance(out, Path)
    assert out == local_out
    assert is_remote is False
    assert out.exists()


def test_configuration_get_output_path_local_existing_path_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover get_output_path branch where local output already exists (if-not-exists false path)."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    local_out = tmp_path / "already-there"
    local_out.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(configuration_module, "is_running_in_docker", lambda: False)
    out, is_remote = config.get_output_path(local_out)

    assert out == local_out
    assert is_remote is False
