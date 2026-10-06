"""Tests for the Config and Metadata class."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from paidiverpy.metadata_parser import ifdo_tools
from paidiverpy.metadata_parser.ifdo_tools import validate_ifdo
from paidiverpy.pipeline.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


def _ifdo_schema() -> dict:
    return {
        "$defs": {
            "image-item-core": {"required": ["image-uuid", "image-datetime"]},
            "uuid": {"pattern": "^[0-9a-f-]{36}$"},
            "iFDO-fields": {"anyOf": [{"$ref": "#/$defs/headerFields"}, {"$ref": "#/$defs/itemFields"}]},
            "headerFields": {
                "properties": {
                    "image-set-name": {"type": "string", "description": "set"},
                    "image-set-handle": {"type": "string", "description": "handle"},
                    "image-datetime": {"type": "string", "description": "dt"},
                    "image-latitude": {"type": "number", "description": 0.0},
                    "image-longitude": {"type": "number", "description": 0.0},
                }
            },
            "itemFields": {
                "properties": {
                    "filename": {"type": "string", "description": "name"},
                    "image-uuid": {"type": "string", "description": "uuid"},
                    "image-datetime": {"type": "string", "description": "dt"},
                    "image-sensor": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "description": "sensor"}},
                        "required": ["name"],
                    },
                }
            },
        },
        "properties": {"image-set-header": {"required": ["image-set-name"]}},
    }


class TestIFDOUtilsClass(BaseTestClass):
    """Tests for the Config and Metadata class.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_validate_ifdo_utils(self):
        """Test the validate_ifdo_utils function."""
        ifdo_path = "tests/example_files/metadata_ifdo.json"
        with Path(ifdo_path).open() as file:
            ifdo_data = json.load(file)
        errors = validate_ifdo(file_path=ifdo_path)
        assert isinstance(errors, list)
        assert len(errors) > 0
        errors = validate_ifdo(ifdo_data=ifdo_data)
        assert isinstance(errors, list)
        assert len(errors) > 0
        with pytest.raises(ValueError) as cm:
            validate_ifdo()
        assert "Either file_path or ifdo_data must be provided" in str(cm.value)

    def test_export_metadata_to_csv_json(self):
        """Test the export_metadata_to_csv function."""
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic.yml")
        pipeline.run()
        pipeline.metadata.export_metadata("json")
        output_metadata_path = Path("./metadata.json")
        assert Path(output_metadata_path).exists()
        output_metadata_path.unlink()
        pipeline.metadata.export_metadata("csv")
        output_metadata_path = Path("./metadata.csv")
        assert Path(output_metadata_path).exists()
        output_metadata_path.unlink()

    def test_export_metadata_to_ifdo(self):
        """Test the export_metadata_to_ifdo function."""
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic.yml")
        pipeline.run()
        pipeline.metadata.export_metadata("ifdo")
        output_metadata_path = Path("./metadata.json")
        assert Path(output_metadata_path).exists()
        errors = validate_ifdo(output_metadata_path.absolute())
        assert isinstance(errors, list)
        assert len(errors) == 0
        output_metadata_path.unlink()

    def test_ifdo_convert_to_ifdo_missing_fields_and_validation_warnings(self):
        """Test the convert_to_ifdo function with missing fields and validation warnings."""
        schema = _ifdo_schema()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(ifdo_tools, "get_file_from_bucket", lambda *_a, **_k: json.dumps(schema).encode("utf-8"))
            monkeypatch.setattr(ifdo_tools, "validate_ifdo", lambda **_k: [{"path": ["a", "b"], "message": "bad"}])

            warnings: list[str] = []
            monkeypatch.setattr(ifdo_tools, "logger", SimpleNamespace(warning=lambda msg, *args: warnings.append(msg % args if args else msg)))

            with tempfile.TemporaryDirectory() as tmpdir:
                metadata = pd.DataFrame({"filename": ["a.jpg"], "ID": ["id-1"]})
                out = Path(tmpdir) / "ifdo.json"
                ifdo_tools.convert_to_ifdo({}, metadata, str(out))
                assert out.exists()
                assert any("Missing required fields" in w for w in warnings)
                assert any("Validation errors" in w for w in warnings)

    def test_ifdo_parse_items_and_header_remaining_branches(self):
        """Test the parse_ifdo_items and parse_ifdo_header functions with remaining branches."""
        schema = _ifdo_schema()

        metadata = pd.DataFrame(
            {
                "filename": ["a.jpg"],
                "ID": ["id-abc"],
                "image-sensor": [{"name": "cam"}],
                "image-latitude": [1.0],
                "image-longitude": [2.0],
                "image-datetime": pd.to_datetime(["2024-01-01T00:00:00"]),
            }
        )

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                ifdo_tools,
                "map_exif_to_ifdo",
                lambda _row: {"image-sensor": {"name": "cam"}, "extra": "x"},
            )

            items, _missing = ifdo_tools.parse_ifdo_items(metadata.copy(), schema)

            assert "a.jpg" in items
            assert items["a.jpg"]["image-uuid"] == "id-abc"

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = str(Path(tmpdir) / "out")
                header_1, _ = ifdo_tools.parse_ifdo_header({"output_path": output_path}, schema, metadata)
                assert header_1["image-set-handle"] == output_path

                input_path = str(Path(tmpdir) / "in")
                header_2, _ = ifdo_tools.parse_ifdo_header(
                    {
                        "input_path": input_path,
                        "image-datetime": "preset",
                        "image-latitude": 1.0,
                        "image-longitude": 2.0,
                    },
                    schema,
                    metadata,
                )
                assert header_2["image-set-handle"] == input_path

    def test_ifdo_parse_validation_errors_invalid_uuid_and_fallback_message(self):
        """Test the parse_validation_errors function with invalid UUID and fallback message."""
        schema = _ifdo_schema()

        errors = [
            {
                "path": ["image-set-items", "x"],
                "message": "{'image-uuid': 'bad', 'image-datetime': '2024-01-01'} is not valid under any of the given schemas",
            }
        ]
        parsed = ifdo_tools.parse_validation_errors(errors, schema)
        assert "Invalid or missing 'image-uuid'" in parsed[0]["message"]

        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        errors_2 = [
            {
                "path": ["image-set-items", "x"],
                "message": f"{{'image-uuid': '{valid_uuid}', 'image-datetime': '2024-01-01'}} is not valid under any of the given schemas",
            }
        ]
        parsed_2 = ifdo_tools.parse_validation_errors(errors_2, schema)
        assert parsed_2[0]["message"].endswith("schemas")

    def test_ifdo_validate_ifdo_file_path_happy_path(self):
        """Test the validate_ifdo function with a valid file path."""
        ifdo_payload = {
            "image-set-header": {"image-set-ifdo-version": "v2.1.0"},
            "image-set-items": {},
        }
        class _FakeValidator:
            def __init__(self, _schema: dict):
                pass

            def iter_errors(self, _data: dict) -> list:
                """Simulate no validation errors.

                Args:
                    _data (dict): The data to validate.

                Returns:
                    list: An empty list indicating no validation errors.
                """
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            ifdo_file = Path(tmpdir) / "ifdo.json"
            ifdo_file.write_text(json.dumps(ifdo_payload), encoding="utf-8")

            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(ifdo_tools, "get_file_from_bucket", lambda *_a, **_k: json.dumps(_ifdo_schema()).encode("utf-8"))
                monkeypatch.setattr(ifdo_tools, "Draft202012Validator", _FakeValidator)

                errors = ifdo_tools.validate_ifdo(file_path=str(ifdo_file))
                assert errors == []



    def test_ifdo_validate_missing_both_params(self):
        """Test validate_ifdo when both file_path and ifdo_data are missing."""
        with pytest.raises(ValueError, match="Either file_path or ifdo_data"):
            ifdo_tools.validate_ifdo()


    def test_ifdo_format_error_message(self):
        """Test format_ifdo_validation_error for short and long paths."""
        short = ifdo_tools.format_ifdo_validation_error(["a", "b"])
        long = ifdo_tools.format_ifdo_validation_error(["x", "y", "z", "w"])
        assert short == "a.b"
        assert long == "...y.z.w"


if __name__ == "__main__":
    unittest.main()
