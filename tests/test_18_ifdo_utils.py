"""Tests for the Config and Metadata class."""

import json
import unittest
from pathlib import Path
import pytest
from paidiverpy.metadata_parser.ifdo_tools import validate_ifdo
from paidiverpy.pipeline.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


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
        assert len(errors) > 0
        output_metadata_path.unlink()


if __name__ == "__main__":
    unittest.main()
