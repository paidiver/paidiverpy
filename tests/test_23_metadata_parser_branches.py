"""Focused tests for remaining branches in metadata_parser.py."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from paidiverpy.metadata_parser import metadata_parser as metadata_module
from paidiverpy.metadata_parser.metadata_parser import MetadataParser
from tests.utils import DummyLogger


def test_metadata_parser_init_loads_from_file_list_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cover __init__ branch where metadata is created from input files."""
    (tmp_path / "a.png").write_bytes(b"1")
    (tmp_path / "b.png").write_bytes(b"2")

    config = SimpleNamespace(
        general=SimpleNamespace(
            input_path=str(tmp_path),
            file_name_pattern="*.png",
            metadata_path=None,
            append_data_to_metadata=None,
            metadata_type=None,
        )
    )

    monkeypatch.setattr(MetadataParser, "_prepare_metadata", lambda _self, metadata: metadata)

    parser = MetadataParser(config=config, use_dask=False)

    assert len(parser.metadata) == 2
    assert "filename" in parser.metadata.columns
    assert "ID" in parser.metadata.columns


def test_metadata_parser_export_metadata_uses_supplied_metadata_and_dataset(tmp_path: Path):
    """Cover export_metadata branches where metadata and dataset_metadata are provided."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata = pd.DataFrame({"filename": ["f1"], "flag": [0]})
    parser.dataset_metadata = {"dataset": "fallback"}

    supplied_metadata = pd.DataFrame({"filename": ["f2"], "flag": [0]})
    supplied_dataset = {"dataset": "explicit"}
    out_path = tmp_path / "meta_out"

    parser.export_metadata(
        output_format="csv",
        output_path=str(out_path),
        metadata=supplied_metadata,
        dataset_metadata=supplied_dataset,
    )

    exported = pd.read_csv(f"{out_path}.csv")
    assert exported["filename"].tolist() == ["f2"]
    assert exported["dataset"].tolist() == ["explicit"]


def test_metadata_parser_export_metadata_exception_path(monkeypatch: pytest.MonkeyPatch):
    """Cover export_metadata exception handling branch."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata = pd.DataFrame({"filename": ["f1"], "flag": [0]})
    parser.dataset_metadata = {"dataset": "d1"}

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(MetadataParser, "convert_metadata_to", staticmethod(_boom))

    with pytest.raises(ValueError, match="Failed to export metadata"):
        parser.export_metadata(output_format="csv", output_path="unused")


def test_metadata_parser_prepare_metadata_logs_aggregate_warning(monkeypatch: pytest.MonkeyPatch):
    """Cover _prepare_metadata branch that emits aggregate warning when columns are missing."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_conventions = {
        "image-altitude-meters": ["alt"],
        "image-depth": ["depth"],
        "image-latitude": ["lat"],
        "image-longitude": ["lon"],
        "image-camera-pitch-degrees": ["pitch"],
        "image-camera-roll-degrees": ["roll"],
    }

    logs: list[str] = []
    monkeypatch.setattr(metadata_module, "logger", SimpleNamespace(warning=lambda msg, *args: logs.append(msg % args if args else msg)))

    out = MetadataParser._prepare_metadata(parser, pd.DataFrame({"filename": ["img.jpg"]}))

    assert "filename" in out.columns
    assert any("Some functions may not work properly." in msg for msg in logs)


def test_open_ifdo_metadata_docker_path_rewrite_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _open_ifdo_metadata docker rewrite branch and successful local JSON load."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "relative/meta_ifdo.json"
    parser.storage_options = {}

    payload = {
        "image-set-header": {"image-set-name": "demo"},
        "image-set-items": {"img.jpg": {"image-datetime": "2024-01-01T00:00:00"}},
    }

    class _FakePath:
        def __init__(self, value: str):
            self._value = value
            self.name = value.split("/")[-1]

        def open(self):
            return io.StringIO(json.dumps(payload))

    monkeypatch.setattr(metadata_module, "Path", _FakePath)
    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: False)
    monkeypatch.setattr(metadata_module, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(parser, "_validate_ifdo", lambda _m: None)

    result = MetadataParser._open_ifdo_metadata(parser)

    assert parser.dataset_metadata["image-set-name"] == "demo"
    assert "filename" in result.columns
    assert "ID" in result.columns


def test_open_ifdo_metadata_json_decode_error_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _open_ifdo_metadata JSONDecodeError branch."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "meta_ifdo.json"
    parser.storage_options = {}

    class _FakePath:
        def __init__(self, value: str):
            self.name = value.split("/")[-1]

        def open(self):
            return io.StringIO("{ invalid json")

    monkeypatch.setattr(metadata_module, "Path", _FakePath)
    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: False)
    monkeypatch.setattr(metadata_module, "is_running_in_docker", lambda: False)

    with pytest.raises(json.JSONDecodeError):
        MetadataParser._open_ifdo_metadata(parser)


def test_open_csv_metadata_remote_bytes_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _open_csv_metadata remote-download branch."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "s3://bucket/metadata.csv"
    parser.storage_options = {"service_name": "s3"}

    csv_bytes = b"filename,ID,image-datetime\na.jpg,id-1,2024-01-01T00:00:00\n"

    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: True)
    monkeypatch.setattr(metadata_module, "get_file_from_bucket", lambda *_a, **_k: csv_bytes)
    monkeypatch.setattr(parser, "_rename_columns", lambda metadata, *_a, **_k: metadata)
    monkeypatch.setattr(parser, "_handle_datetime", lambda metadata: metadata)

    out = MetadataParser._open_csv_metadata(parser)

    assert out["filename"].tolist() == ["a.jpg"]
    assert out["ID"].tolist() == ["id-1"]


def test_open_csv_metadata_docker_rewrite_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _open_csv_metadata docker metadata-path rewrite branch."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "local/meta.csv"
    parser.storage_options = {}
    parser.config = SimpleNamespace(general=SimpleNamespace(sample_data=None))

    calls: list[str] = []

    def _fake_read_csv(path):
        calls.append(str(path))
        return pd.DataFrame({"filename": ["a.jpg"], "image-datetime": ["2024-01-01T00:00:00"]})

    def _rename(metadata, column_name, *_args, **_kwargs):
        if column_name == "ID":
            raise ValueError("missing id")
        return metadata

    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: False)
    monkeypatch.setattr(metadata_module, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(metadata_module.pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr(parser, "_rename_columns", _rename)
    monkeypatch.setattr(parser, "_handle_datetime", lambda metadata: metadata)

    out = MetadataParser._open_csv_metadata(parser)

    assert calls == ["/app/metadata/meta.csv"]
    assert "ID" in out.columns


def test_validate_ifdo_success_logs_info(monkeypatch: pytest.MonkeyPatch):
    """Cover _validate_ifdo success branch."""
    parser = MetadataParser.__new__(MetadataParser)
    messages: list[str] = []

    monkeypatch.setattr(metadata_module, "validate_ifdo", lambda **_k: [])
    monkeypatch.setattr(metadata_module, "logger", SimpleNamespace(info=lambda msg, *args: messages.append(msg % args if args else msg)))

    MetadataParser._validate_ifdo(parser, {"image-set-header": {}, "image-set-items": {}})

    assert any("Metadata file is valid." in m for m in messages)


def test_compute_with_dask_dataframe_branch():
    """Cover compute() branch when metadata is a Dask DataFrame."""
    dd = pytest.importorskip("dask.dataframe")

    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata = dd.from_pandas(pd.DataFrame({"a": [1, 2]}), npartitions=1)

    parser.compute()

    assert isinstance(parser.metadata, pd.DataFrame)
    assert parser.metadata["a"].tolist() == [1, 2]


def test_convert_metadata_to_unsupported_else_branch(tmp_path: Path):
    """Cover convert_metadata_to final unsupported-format else branch."""
    metadata = pd.DataFrame({"filename": ["a.jpg"]})

    with pytest.raises(ValueError, match="Unsupported output format"):
        MetadataParser.convert_metadata_to({}, metadata, str(tmp_path / "out"), "xml")


def test_group_metadata_branch_when_key_already_exists():
    """Cover group_metadata_and_dataset_metadata branch where key already exists."""
    metadata = pd.DataFrame({"dataset": ["existing"], "filename": ["a.jpg"]})
    grouped = MetadataParser.group_metadata_and_dataset_metadata(metadata.copy(), {"dataset": "new"})

    assert grouped["dataset"].tolist() == ["existing"]
