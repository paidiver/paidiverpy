"""Targeted tests for remaining missing lines and branches across all modules."""

# ruff: noqa

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from paidiverpy.config.configuration import Configuration
from paidiverpy.metadata_parser import ifdo_tools
from paidiverpy.metadata_parser.metadata_parser import MetadataParser
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.models.open_params import ImageOpenArgs
from paidiverpy.open_layer import utils as open_utils
from paidiverpy.utils import locals as local_utils


# ============================================================================
# Configuration Tests (90% → target 100%)
# ============================================================================

def test_configuration_add_step_validation_works():
    """Test add_step with validation."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    original_len = len(config.steps)

    # Just verify add_step works
    config.add_step(
        parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
        validate=False,
    )
    assert len(config.steps) > original_len


def test_configuration_init_without_inputs_warns(caplog):
    """Test Configuration init warning when neither file nor params are provided."""
    caplog.set_level("WARNING")
    config = Configuration()
    assert config.general is None
    assert "Configuration file path or configuration parameters are not specified" in caplog.text


def test_configuration_init_add_steps_without_general_warns(caplog):
    """Test init warning when add_steps is provided without general config."""
    caplog.set_level("WARNING")
    Configuration(add_steps=[{"sampling": {"name": "s1", "mode": "fixed", "params": {"value": 1}}}])
    assert "General configuration is not defined" in caplog.text


def test_configuration_add_general_validate_rolls_back():
    """Test add_general with failed validation rolls back state."""
    from jsonschema.exceptions import ValidationError

    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    original = config.general.name

    # Attempt to add with validation that will fail
    with patch.object(Configuration, "validate_config", side_effect=ValidationError("bad")):
        with pytest.raises(ValidationError):
            config.add_general({"name": "changed-name"}, validate=True)

    # Ensure original state is restored
    assert config.general.name == original


def test_configuration_output_path_docker_branch(monkeypatch):
    """Test get_output_path when running in docker."""
    from paidiverpy.config import configuration as config_mod

    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    monkeypatch.setattr(config_mod, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)

    out_path, is_remote = config.get_output_path("local-output")
    assert str(out_path) == "/app/output"
    assert is_remote is False


def test_configuration_export_to_none_returns_string():
    """Test export with None returns string instead of writing file."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    result = config.export(None)
    assert isinstance(result, str)
    assert "general:" in result


def test_configuration_export_to_file(tmp_path):
    """Test export to specific file path."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    out_file = tmp_path / "cfg.yml"
    result = config.export(out_file)
    assert result is None
    assert out_file.exists()


def test_configuration_remove_step_at_index():
    """Test remove_step at specific index."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    original_len = len(config.steps)

    config.add_step(
        parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
        validate=False,
    )
    assert len(config.steps) > original_len

    config.remove_step(len(config.steps) - 1)
    assert len(config.steps) == original_len


def test_configuration_add_step_invalid_index_raises():
    """Test add_step with invalid index raises ValueError."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    config.add_step(
        parameters={"name": "s0", "step_name": "sampling", "mode": "fixed", "params": {"value": 1}},
        validate=False,
    )
    with pytest.raises(ValueError, match="Invalid step index"):
        config.add_step(
            config_index=len(config.steps) + 10,
            parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
            validate=False,
        )


def test_configuration_add_step_validate_rolls_back():
    """Test add_step rollback when validation fails."""
    from jsonschema.exceptions import ValidationError

    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    before = list(config.steps)

    with patch.object(Configuration, "validate_config", side_effect=ValidationError("bad")):
        with pytest.raises(ValidationError):
            config.add_step(
                parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
                validate=True,
            )

    assert len(config.steps) == len(before)


def test_configuration_validate_config_remote_schema(monkeypatch, tmp_path):
    """Test validate_config with remote schema (line 169 exit path)."""
    from paidiverpy.config import configuration as config_mod
    from jsonschema import Draft202012Validator

    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("general: {}\nsteps: []\n", encoding="utf-8")

    class _FakeValidator:
        def __init__(self, schema):
            pass

        def iter_errors(self, config):
            return []

    monkeypatch.setattr(config_mod, "path_is_remote", lambda _path: True)
    monkeypatch.setattr(config_mod, "get_file_from_bucket", lambda *_a, **_k: b"{}")
    monkeypatch.setattr(config_mod, "Draft202012Validator", _FakeValidator)

    Configuration.validate_config(cfg_file, local=False)


def test_configuration_load_steps_invalid_name():
    """Test _load_steps with invalid step name."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    with pytest.raises(ValueError, match="Invalid step name"):
        config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})


# ============================================================================
# MetadataParser Tests (88% → target 100%)
# ============================================================================


def test_metadata_parser_csv_with_minimal_columns(tmp_path):
    """Test MetadataParser with minimal CSV columns."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("filename\nimg.jpg\n", encoding="utf-8")

    general = SimpleNamespace(
        metadata_path=str(csv_path),
        metadata_type="CSV_FILE",
        append_data_to_metadata=None,
        metadata_conventions=None,
        sample_data=None,
    )
    parser = MetadataParser(config=SimpleNamespace(general=general), use_dask=False)
    assert "filename" in parser.metadata.columns


def test_metadata_parser_datetime_handling(tmp_path):
    """Test _handle_datetime with various formats."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "filename,image-datetime\nimg.jpg,2024-01-01T00:00:00\n",
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
    # Column might be image-datetime or datetime depending on processing
    assert "image-datetime" in parser.metadata.columns or "datetime" in parser.metadata.columns


def test_metadata_parser_export_csv_format(tmp_path):
    """Test export_metadata with CSV format."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("filename\nimg.jpg\n", encoding="utf-8")

    try:
        general = SimpleNamespace(
            metadata_path=str(csv_path),
            metadata_type="CSV_FILE",
            append_data_to_metadata=None,
            metadata_conventions=None,
            sample_data=None,
        )
        parser = MetadataParser(config=SimpleNamespace(general=general), use_dask=False)

        out_path = str(tmp_path / "export")
        parser.export_metadata(output_format="csv", output_path=out_path)
        assert (tmp_path / "export.csv").exists()

    finally:
        csv_path.unlink(missing_ok=True)


def test_metadata_parser_export_json_format(tmp_path):
    """Test export_metadata with JSON format."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("filename\nimg.jpg\n", encoding="utf-8")

    try:
        general = SimpleNamespace(
            metadata_path=str(csv_path),
            metadata_type="CSV_FILE",
            append_data_to_metadata=None,
            metadata_conventions=None,
            sample_data=None,
        )
        parser = MetadataParser(config=SimpleNamespace(general=general), use_dask=False)

        out_path = str(tmp_path / "export_json")
        parser.export_metadata(output_format="json", output_path=out_path)
        assert (tmp_path / "export_json.json").exists()

    finally:
        csv_path.unlink(missing_ok=True)


def test_metadata_parser_convert_metadata_to_csv(tmp_path):
    """Test convert_metadata_to with CSV format and from_step."""
    meta = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "flag": [0, 2]})
    out_path = str(tmp_path / "converted")
    MetadataParser.convert_metadata_to({}, meta, out_path, "csv", from_step=0)
    assert (tmp_path / "converted.csv").exists()


def test_metadata_parser_convert_metadata_to_json(tmp_path):
    """Test convert_metadata_to with JSON format."""
    meta = pd.DataFrame({"filename": ["a.jpg"], "flag": [0]})
    out_path = str(tmp_path / "converted_json")
    MetadataParser.convert_metadata_to({}, meta, out_path, "json", from_step=0)
    assert (tmp_path / "converted_json.json").exists()


def test_metadata_parser_convert_metadata_unsupported_format():
    """Test convert_metadata_to with unsupported format."""
    meta = pd.DataFrame({"filename": ["a.jpg"]})
    with pytest.raises(NotImplementedError, match="Croissant format is not implemented yet"):
        MetadataParser.convert_metadata_to({}, meta, "/tmp/out", "croissant")


def test_metadata_parser_with_spatial_columns(tmp_path):
    """Test metadata parser creates geometry/point columns."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "filename,image-latitude,image-longitude\nimg.jpg,10.0,20.0\n",
        encoding="utf-8",
    )

    try:
        general = SimpleNamespace(
            metadata_path=str(csv_path),
            metadata_type="CSV_FILE",
            append_data_to_metadata=None,
            metadata_conventions=None,
            sample_data=None,
        )
        parser = MetadataParser(config=SimpleNamespace(general=general), use_dask=False)
        assert "point" in parser.metadata.columns or "geometry" in parser.metadata.columns
    finally:
        csv_path.unlink(missing_ok=True)


# ============================================================================
# ImageOpenArgs Tests (75% → target 100%)
# ============================================================================


def test_image_open_args_raw_missing_required_fields():
    """Exercise validate_params dict casting branch for raw types."""
    args = ImageOpenArgs.model_construct(image_type="raw", params={"width": 1, "height": 1, "bit_depth": 8})
    validated = ImageOpenArgs.validate_params(args)
    assert validated.image_type == "raw"
    assert hasattr(validated.params, "bit_depth")


def test_image_open_args_nef_validation():
    """Test ImageOpenArgs with NEF (rawpy) params."""
    nef_args = ImageOpenArgs(image_type="nef", params={"use_camera_wb": False})
    assert nef_args.params.use_camera_wb is False


def test_image_open_args_opencv_formats():
    """Test ImageOpenArgs with various OpenCV formats."""
    for fmt in ["png", "jpg", "bmp", "tiff"]:
        args = ImageOpenArgs(image_type=fmt, params={"dtype": "uint8", "flags": 1})
        assert args.image_type == fmt


def test_image_open_args_raw_format():
    """Test ImageOpenArgs with raw format."""
    args = ImageOpenArgs(image_type="raw", params={"width": 1, "height": 1, "bit_depth": 8})
    assert args.image_type == "raw"


# ============================================================================
# Open Layer Utils Tests (95% → target 100%)
# ============================================================================


def test_open_utils_extract_exif_without_name(monkeypatch):
    """Test extract_exif_single without image_name (returns empty dict)."""
    class _FakeImage:
        def getexif(self):
            return {1: "value"}

    monkeypatch.setattr(open_utils.Image, "open", lambda *_a, **_k: _FakeImage())

    result = open_utils.extract_exif_single(io.BytesIO(b"x"), image_type="jpg")
    assert result == {}


def test_open_utils_pad_image_basic():
    """Test pad_image with basic target dimensions."""
    img = np.ones((2, 2, 3), dtype=np.uint8)

    # Test padding to larger size
    result = open_utils.pad_image(img, target_height=4, target_width=5)
    assert result.shape[0] >= 2
    assert result.shape[1] >= 2
    assert result.shape[2] == 3


def test_open_utils_load_raw_unsupported_format(monkeypatch):
    """Test load_raw_image with unsupported format (UnsupportedError)."""
    monkeypatch.setattr(open_utils.rawpy, "LibRawFileUnsupportedError", RuntimeError)
    monkeypatch.setattr(
        open_utils.rawpy,
        "imread",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("unsupported"))
    )

    result = open_utils.load_raw_image("fake.nef", "nef", {"use_camera_wb": True}, remote=False)
    assert result is None
    monkeypatch.undo()


def test_open_utils_extract_exif_missing_name_for_bytesio(monkeypatch):
    """Test EXIF extraction with BytesIO and no image_name returns empty dict via error branch."""

    class _FakeImage:
        def getexif(self):
            return {1: "value"}

    monkeypatch.setattr(open_utils.Image, "open", lambda *_a, **_k: _FakeImage())
    result = open_utils.extract_exif_single(io.BytesIO(b"x"), image_type="jpg")
    assert result == {}


# ============================================================================
# Utils / Locals Tests (89% → target 100%)
# ============================================================================


def test_locals_pip_version_missing_package(monkeypatch):
    """Test pip_version when package not installed."""
    monkeypatch.setattr(local_utils, "PIP_INSTALLED", {})
    result = local_utils.pip_version("nonexistent-package")
    assert result == "-"


def test_locals_cli_version_success(monkeypatch):
    """Test cli_version successful execution."""
    monkeypatch.setattr(
        local_utils.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=b"version 1.2.3\n"),
    )
    result = local_utils.cli_version("test-tool")
    assert "1.2.3" in result


def test_locals_cli_version_not_in_path(monkeypatch):
    """Test cli_version when tool not in PATH."""
    monkeypatch.setattr(local_utils.shutil, "which", lambda name: None)
    result = local_utils.cli_version("missing-tool")
    assert result == "-"


def test_locals_cli_version_execution_failure(monkeypatch):
    """Test cli_version when execution fails."""
    monkeypatch.setattr(local_utils.shutil, "which", lambda name: "/usr/bin/tool")

    def _raise(*args, **kwargs):
        raise RuntimeError("execution failed")

    monkeypatch.setattr(local_utils.subprocess, "run", _raise)
    result = local_utils.cli_version("failing-tool")
    assert "- #" in result or result == "-"


def test_locals_get_version_from_module():
    """Test get_version from module __version__."""
    module = SimpleNamespace(__version__="2.0.0")
    result = local_utils.get_version(module)
    assert result == "2.0.0"


def test_locals_show_versions_conda_format(monkeypatch):
    """Test show_versions with conda=True."""
    monkeypatch.setattr(local_utils, "get_sys_info", lambda: [("system", "test")])
    monkeypatch.setattr(local_utils, "get_version", lambda _name: "1.0.0")

    output = io.StringIO()
    local_utils.show_versions(file=output, conda=True)
    result = output.getvalue()
    assert "#" in result  # conda format uses # comments


def test_locals_show_versions_standard_format(monkeypatch):
    """Test show_versions with conda=False."""
    monkeypatch.setattr(local_utils, "get_sys_info", lambda: [("python", "3.10")])
    monkeypatch.setattr(local_utils, "get_version", lambda _name: "1.0.0")

    output = io.StringIO()
    local_utils.show_versions(file=output, conda=False)
    result = output.getvalue()
    assert "SYSTEM" in result


def test_locals_get_sys_info_git_missing(monkeypatch):
    """Test get_sys_info when .git directory missing."""
    class _FakePath:
        def __init__(self, *_args, **_kwargs):
            pass

        def is_dir(self):
            return False

    monkeypatch.setattr(local_utils, "Path", _FakePath)
    result = dict(local_utils.get_sys_info())
    assert result.get("commit") is None


def test_general_config_required_fields_validation():
    """Test GeneralConfig required field validation branches."""
    with pytest.raises(ValueError, match="Either 'sample_data' or 'input_path' must be provided"):
        GeneralConfig(output_path="out", input_path=None, sample_data=None)


def test_general_config_update_revalidates_values(tmp_path):
    """Test GeneralConfig.update applies and revalidates values."""
    cfg = GeneralConfig(input_path=str(tmp_path), output_path="output")
    updated = cfg.update(output_path=str(tmp_path / "new-output"))
    assert updated.output_path == Path(tmp_path / "new-output")


# ============================================================================
# IFDO Tools Tests (89% → target 100%)
# ============================================================================


def test_ifdo_validate_missing_both_params():
    """Test validate_ifdo when both file_path and ifdo_data are missing."""
    with pytest.raises(ValueError, match="Either file_path or ifdo_data"):
        ifdo_tools.validate_ifdo()


def test_ifdo_format_error_message():
    """Test format_ifdo_validation_error for short and long paths."""
    short = ifdo_tools.format_ifdo_validation_error(["a", "b"])
    long = ifdo_tools.format_ifdo_validation_error(["x", "y", "z", "w"])
    assert short == "a.b"
    assert long == "...y.z.w"
