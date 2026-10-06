"""Focused tests for remaining branches in metadata_parser.py."""

from __future__ import annotations
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import NoReturn
import pandas as pd
import pytest
from paidiverpy.metadata_parser import metadata_parser as metadata_module
from paidiverpy.metadata_parser.metadata_parser import MetadataParser
from tests.utils import FakePath

if TYPE_CHECKING:
    from pathlib import Path


def test_metadata_parser_init_loads_from_file_list_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cover __init__ branch where metadata is created from input files."""
    (tmp_path / "a.png").write_bytes(b"1")
    (tmp_path / "b.png").write_bytes(b"2")
    num_images = 2

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

    assert len(parser.metadata) == num_images
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

    def _boom(*_args: tuple, **_kwargs: dict) -> NoReturn:
        msg = "forced failure"
        raise RuntimeError(msg)

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

    out = MetadataParser._prepare_metadata(parser, pd.DataFrame({"filename": ["img.jpg"]}))  # noqa: SLF001

    assert "filename" in out.columns
    assert any("Some functions may not work properly." in msg for msg in logs)


def test_open_ifdo_metadata_json_decode_error_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _open_ifdo_metadata JSONDecodeError branch."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "meta_ifdo.json"
    parser.storage_options = {}


    monkeypatch.setattr(metadata_module, "Path", FakePath)
    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: False)
    monkeypatch.setattr(metadata_module, "is_running_in_docker", lambda: False)

    with pytest.raises(json.JSONDecodeError):
        MetadataParser._open_ifdo_metadata(parser)  # noqa: SLF001


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

    out = MetadataParser._open_csv_metadata(parser)  # noqa: SLF001

    assert out["filename"].tolist() == ["a.jpg"]
    assert out["ID"].tolist() == ["id-1"]


def test_open_csv_metadata_docker_rewrite_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover _open_csv_metadata docker metadata-path rewrite branch."""
    parser = MetadataParser.__new__(MetadataParser)
    parser.metadata_path = "local/meta.csv"
    parser.storage_options = {}
    parser.config = SimpleNamespace(general=SimpleNamespace(sample_data=None))

    calls: list[str] = []

    def _fake_read_csv(path: str) -> pd.DataFrame:
        calls.append(str(path))
        return pd.DataFrame({"filename": ["a.jpg"], "image-datetime": ["2024-01-01T00:00:00"]})

    def _rename(metadata: pd.DataFrame, column_name: str, *_args: tuple, **_kwargs: dict) -> pd.DataFrame:
        if column_name == "ID":
            msg = "missing id"
            raise ValueError(msg)
        return metadata

    monkeypatch.setattr(metadata_module, "path_is_remote", lambda _path: False)
    monkeypatch.setattr(metadata_module, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(metadata_module.pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr(parser, "_rename_columns", _rename)
    monkeypatch.setattr(parser, "_handle_datetime", lambda metadata: metadata)

    out = MetadataParser._open_csv_metadata(parser)  # noqa: SLF001

    assert calls == ["/app/metadata/meta.csv"]
    assert "ID" in out.columns


def test_validate_ifdo_success_logs_info(monkeypatch: pytest.MonkeyPatch):
    """Cover _validate_ifdo success branch."""
    parser = MetadataParser.__new__(MetadataParser)
    messages: list[str] = []

    monkeypatch.setattr(metadata_module, "validate_ifdo", lambda **_k: [])
    monkeypatch.setattr(metadata_module, "logger", SimpleNamespace(info=lambda msg, *args: messages.append(msg % args if args else msg)))

    MetadataParser._validate_ifdo(parser, {"image-set-header": {}, "image-set-items": {}})  # noqa: SLF001

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




def test_metadata_parser_csv_with_minimal_columns(tmp_path: Path):
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


def test_metadata_parser_datetime_handling(tmp_path: Path):
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
    assert "image-datetime" in parser.metadata.columns or "datetime" in parser.metadata.columns


def test_metadata_parser_export_csv_format(tmp_path: Path):
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


def test_metadata_parser_export_json_format(tmp_path: Path):
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


def test_metadata_parser_convert_metadata_to_csv(tmp_path: Path):
    """Test convert_metadata_to with CSV format and from_step."""
    meta = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "flag": [0, 2]})
    out_path = str(tmp_path / "converted")
    MetadataParser.convert_metadata_to({}, meta, out_path, "csv", from_step=0)
    assert (tmp_path / "converted.csv").exists()


def test_metadata_parser_convert_metadata_to_json(tmp_path: Path):
    """Test convert_metadata_to with JSON format."""
    meta = pd.DataFrame({"filename": ["a.jpg"], "flag": [0]})
    out_path = str(tmp_path / "converted_json")
    MetadataParser.convert_metadata_to({}, meta, out_path, "json", from_step=0)
    assert (tmp_path / "converted_json.json").exists()


def test_metadata_parser_convert_metadata_unsupported_format(tmp_path: Path):
    """Test convert_metadata_to with unsupported format."""
    meta = pd.DataFrame({"filename": ["a.jpg"]})
    with pytest.raises(NotImplementedError, match="Croissant format is not implemented yet"):
        MetadataParser.convert_metadata_to({}, meta,  str(tmp_path / "out"), "croissant")


def test_metadata_parser_with_spatial_columns(tmp_path: Path):
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
