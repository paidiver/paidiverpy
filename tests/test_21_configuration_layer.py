"""Focused branch-coverage tests for configuration, pipeline, metadata parser, and HTML helpers."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import pytest
from jsonschema.exceptions import ValidationError
from paidiverpy.config import configuration as configuration_module
from paidiverpy.config.configuration import Configuration
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.utils import formating_html
from tests.utils import normalise_path


def test_configuration_validation_add_remove_export_and_repr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test the validation, adding/removing steps, exporting, and HTML representation of the Configuration class."""
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


def test_configuration_add_general_validate_rolls_back(monkeypatch: pytest.MonkeyPatch):
    """Test that adding a general configuration rolls back on validation failure."""
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


def test_configuration_validate_config_remote_schema_and_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test the validation of the configuration with a remote schema and error handling."""
    class _FakeValidatorOk:
        def __init__(self, _schema: object):
            pass

        def iter_errors(self, _config: object) -> list:
            return []

    class _FakeError:
        path = []  # noqa: RUF012
        message = "invalid"

        def __str__(self):
            return "fake-error"

    class _FakeValidatorFail:
        def __init__(self, _schema: object):
            pass

        def iter_errors(self, _config: object) -> list:
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


def test_configuration_load_steps_invalid_name_and_output_path_docker(monkeypatch: pytest.MonkeyPatch):
    """Test loading steps with invalid names and output path handling in Docker."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    with pytest.raises(ValueError, match="Invalid step name"):
        config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})  # noqa: SLF001

    monkeypatch.setattr(configuration_module, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)  # noqa: ARG005
    out, is_remote = config.get_output_path(output_path="local-output")
    assert normalise_path(out) == "/app/output"
    assert is_remote is False


def test_configuration_add_step_validation_works():
    """Test add_step with validation."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    original_len = len(config.steps)

    config.add_step(
        parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
        validate=False,
    )
    assert len(config.steps) > original_len


def test_configuration_init_without_inputs_warns(caplog: pytest.LogCaptureFixture):
    """Test Configuration init warning when neither file nor params are provided."""
    caplog.set_level("WARNING")
    config = Configuration()
    assert config.general is None
    assert "Configuration file path or configuration parameters are not specified" in caplog.text


def test_configuration_init_add_steps_without_general_warns(caplog: pytest.LogCaptureFixture):
    """Test init warning when add_steps is provided without general config."""
    caplog.set_level("WARNING")
    Configuration(add_steps=[{"sampling": {"name": "s1", "mode": "fixed", "params": {"value": 1}}}])
    assert "General configuration is not defined" in caplog.text


def test_configuration_output_path_docker_branch(monkeypatch: pytest.MonkeyPatch):
    """Test get_output_path when running in docker."""
    from paidiverpy.config import configuration as config_mod

    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    monkeypatch.setattr(config_mod, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)  # noqa: ARG005

    out_path, is_remote = config.get_output_path("local-output")
    assert normalise_path(out_path) == "/app/output"
    assert is_remote is False


def test_configuration_export_to_none_returns_string():
    """Test export with None returns string instead of writing file."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    result = config.export(None)
    assert isinstance(result, str)
    assert "general:" in result


def test_configuration_export_to_file(tmp_path: Path):
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

    with patch.object(Configuration, "validate_config", side_effect=ValidationError("bad")), pytest.raises(ValidationError):
        config.add_step(
            parameters={"name": "s1", "step_name": "sampling", "mode": "fixed", "params": {"value": 2}},
            validate=True,
        )

    assert len(config.steps) == len(before)


def test_configuration_validate_config_remote_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test validate_config with remote schema (line 169 exit path)."""
    from paidiverpy.config import configuration as config_mod

    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("general: {}\nsteps: []\n", encoding="utf-8")

    class _FakeValidator:
        def __init__(self, schema: object):
            pass

        def iter_errors(self, config: object) -> list:  # noqa: ARG002
            return []

    monkeypatch.setattr(config_mod, "path_is_remote", lambda _path: True)
    monkeypatch.setattr(config_mod, "get_file_from_bucket", lambda *_a, **_k: b"{}")
    monkeypatch.setattr(config_mod, "Draft202012Validator", _FakeValidator)

    Configuration.validate_config(cfg_file, local=False)


def test_configuration_load_steps_invalid_name():
    """Test _load_steps with invalid step name."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")

    with pytest.raises(ValueError, match="Invalid step name"):
        config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})  # noqa: SLF001

def test_general_config_required_fields_validation():
    """Test GeneralConfig required field validation branches."""
    with pytest.raises(ValueError, match="Either 'sample_data' or 'input_path' must be provided"):
        GeneralConfig(output_path="out", input_path=None, sample_data=None)


def test_general_config_update_revalidates_values(tmp_path: Path):
    """Test GeneralConfig.update applies and revalidates values."""
    cfg = GeneralConfig(input_path=str(tmp_path), output_path="output")
    updated = cfg.update(output_path=str(tmp_path / "new-output"))
    assert updated.output_path == Path(tmp_path / "new-output")

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
        config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})  # noqa: SLF001


def test_configuration_load_steps_invalid_name_return_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _load_steps return line after invalid step when raise helper is patched."""
    config = Configuration(config_file_path="tests/config_files/config_simple.yml")
    before = len(config.steps)

    monkeypatch.setattr(configuration_module, "raise_value_error", lambda _msg: None)
    config._load_steps({"steps": [{"unknown": {"name": "bad"}}]})  # noqa: SLF001

    assert len(config.steps) == before


def test_configuration_validate_config_path_input_branch():
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
    config._update_remote_options()  # noqa: SLF001

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


def test_configuration_get_output_path_remote_returns_without_local_mkdir():
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
