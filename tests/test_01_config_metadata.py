"""Tests for the Config and Metadata class."""

import unittest
from json import JSONDecodeError
from pathlib import Path
import pandas as pd
import pytest
import yaml
from jsonschema.exceptions import ValidationError
from paidiverpy import Paidiverpy
from paidiverpy.config.configuration import Configuration
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.open_layer import OpenLayer
from paidiverpy.utils.data import PaidiverpyData
from tests.base_test_class import BaseTestClass


class TestConfigMetadataClass(BaseTestClass):
    """Tests for the Config and Metadata class.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_config_class(self):
        """Test the Config class."""
        config = Configuration(config_file_path="tests/config_files/config_simple.yml")
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        parameters = {"name": "test", "step_name": "sampling", "mode": "overlapping"}
        config.add_step(parameters=parameters)
        with pytest.raises(ValueError) as cm:
            config.add_step(10, parameters=parameters)
        assert "Invalid step index:" in str(cm.value)

    def test_config_class_warning(self):
        """Test the Config class."""
        config = Configuration()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict == {"steps": []}
        parameters = {"name": "test", "step_name": "sampling", "mode": "overlapping"}
        config = Configuration(add_steps=[parameters])
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert len(config_dict["steps"]) == 0
        config_str = config.__repr__()
        assert isinstance(config_str, str)
        data = PaidiverpyData()
        config_params = data.load("plankton_csv")
        config_params.pop("metadata_type")
        with pytest.raises(ValidationError) as cm:
            config = Configuration(add_general=config_params)
        assert "Failed to validate the configuration file" in str(cm.value)

    def test_config_class_errors(self):
        """Test the Config class."""
        with pytest.raises(FileNotFoundError) as cm:
            Configuration(config_file_path="tests/config_files/config_simple_no_exist.yml")
        assert "Failed to load the configuration file" in str(cm.value)
        with pytest.raises(ValidationError) as cm:
            Configuration(config_file_path="tests/config_files/config_simple_error1.yml")
        assert "Failed to validate the configuration file" in str(cm.value)
        with pytest.raises(yaml.YAMLError) as cm:
            Configuration(config_file_path="tests/config_files/config_simple_error2.yml")
        assert "Failed to load the configuration file" in str(cm.value)

    def test_parsing_config_file(self):
        """Test the parsing of the configuration file."""
        classes = [Paidiverpy, OpenLayer]
        for class_name in classes:
            paidiver = class_name(config_file_path="tests/config_files/config_simple.yml")
            self.check_config(paidiver)

    def test_parsing_metadata(self):
        """Test the parsing of the configuration file."""
        config = Configuration(config_file_path="tests/config_files/config_simple.yml")
        metadata = MetadataParser(config=config)
        assert isinstance(metadata, MetadataParser)

    def check_config(self, paidiver: Paidiverpy):
        """Check the configuration file.

        Args:
            paidiver (Paidiverpy): The paidiver object.
        """
        assert isinstance(paidiver.config, Configuration)
        general = paidiver.config.general
        assert general.input_path == (Path.home() / ".paidiverpy_cache/benthic_csv/images").absolute()
        assert str(general.output_path) == "output"
        assert len(general.sampling) > 0
        steps = paidiver.config.steps
        assert len(steps) == 0
        metadata = paidiver.get_metadata()
        assert isinstance(metadata, pd.DataFrame)

    def test_metadata_conventions(self):
        """Test the metadata conventions."""
        file_path = "tests/config_files/config_simple.yml"
        output_file_path = "tests/config_files/config_simple_with_conventions.yml"
        metadata_conventions_path = "tests/example_files/metadata_conventions.json"
        with Path(file_path).open() as file:
            data = yaml.safe_load(file)
        data["general"]["metadata_conventions"] = str(Path(metadata_conventions_path).absolute())
        with Path(output_file_path).open("w") as file:
            yaml.dump(data, file, sort_keys=False)

        config = Configuration(config_file_path=output_file_path)
        metadata = MetadataParser(config=config)
        assert isinstance(metadata, MetadataParser)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert isinstance(config_dict["general"]["metadata_conventions"], str)
        assert config_dict["general"]["metadata_conventions"] == str(Path(metadata_conventions_path).absolute())
        Path(output_file_path).unlink(missing_ok=True)
        metadata_conventions_path_error = "tests/example_files/metadata_conventions_missing.json"
        with Path(file_path).open() as file:
            data = yaml.safe_load(file)
        data["general"]["metadata_conventions"] = str(Path(metadata_conventions_path_error).absolute())
        with Path(output_file_path).open("w") as file:
            yaml.dump(data, file, sort_keys=False)
        config = Configuration(config_file_path=output_file_path)
        metadata = MetadataParser(config=config)
        assert isinstance(metadata, MetadataParser)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert isinstance(config_dict["general"]["metadata_conventions"], str)
        assert config_dict["general"]["metadata_conventions"] == str(Path(metadata_conventions_path_error).absolute())
        Path(output_file_path).unlink(missing_ok=True)

    def test_metadata_conventions_error(self):
        """Test the metadata conventions."""
        file_path = "tests/config_files/config_simple.yml"
        output_file_path = "tests/config_files/config_simple_with_conventions.yml"
        metadata_conventions_path = "tests/example_files/metadata_conventions_error.json"
        with Path(file_path).open() as file:
            data = yaml.safe_load(file)
        data["general"]["metadata_conventions"] = str(Path(metadata_conventions_path).absolute())
        with Path(output_file_path).open("w") as file:
            yaml.dump(data, file, sort_keys=False)

        config = Configuration(config_file_path=output_file_path)
        with pytest.raises(ValueError) as cm:
            MetadataParser(config=config)
        assert "Column image-depth is not in the metadata conventions file" in str(cm.value)

    def test_build_metadata(self):
        """Test the metadata conventions."""
        config = Configuration(config_file_path="tests/config_files/config_simple.yml")
        config_dict_general = config.to_dict()["general"]
        metadata_params = {
            "metadata_conventions": config_dict_general.get("metadata_conventions"),
            "metadata_path": config_dict_general["metadata_path"],
            "metadata_type": config_dict_general["metadata_type"],
            "append_data_to_metadata": config_dict_general["append_data_to_metadata"],
        }
        metadata = MetadataParser(**metadata_params)
        assert isinstance(metadata, MetadataParser)
        metadata_str = metadata.__repr__()
        assert isinstance(metadata_str, str)
        metadata_html = metadata._repr_html_()
        assert isinstance(metadata_html, str)

    def test_metadata_error(self):
        """Test the metadata conventions."""
        config = Configuration(config_file_path="tests/config_files/config_simple.yml")
        config_dict_general = config.to_dict()["general"]
        metadata_params = {
            "metadata_conventions": config_dict_general.get("metadata_conventions"),
            "metadata_path": "tests/example_files/metadata_error.csv",
            "metadata_type": config_dict_general["metadata_type"],
            "append_data_to_metadata": config_dict_general["append_data_to_metadata"],
        }
        with pytest.raises(ValueError) as cm:
            MetadataParser(**metadata_params)
        assert "Metadata does not have a" in str(cm.value)

        metadata_params = {
            "metadata_conventions": config_dict_general.get("metadata_conventions"),
            "metadata_path": config_dict_general["metadata_path"],
            "metadata_type": config_dict_general["metadata_type"],
            "append_data_to_metadata": "tests/example_files/appended_metadata_benthic_csv_error.csv",
        }
        metadata = MetadataParser(**metadata_params)
        metadata_df = metadata.metadata
        assert isinstance(metadata_df, pd.DataFrame)
        assert "pressure_psi" not in metadata_df.columns

    def test_metadata_error_ifdo(self):
        """Test the metadata conventions."""
        config = Configuration(config_file_path="tests/config_files/config_benthic_ifdo.yml")
        config_dict_general = config.to_dict()["general"]
        metadata_params = {
            "metadata_conventions": config_dict_general.get("metadata_conventions"),
            "metadata_path": "tests/example_files/metadata_no_exist.csv",
            "metadata_type": config_dict_general["metadata_type"],
        }
        with pytest.raises(FileNotFoundError) as cm:
            MetadataParser(**metadata_params)
        assert "Metadata file not found" in str(cm.value)
        metadata_params = {
            "metadata_conventions": config_dict_general.get("metadata_conventions"),
            "metadata_path": "tests/example_files/metadata_ifdo_error.json",
            "metadata_type": config_dict_general["metadata_type"],
        }
        with pytest.raises(JSONDecodeError) as cm:
            MetadataParser(**metadata_params)
        assert "Metadata file is not a valid JSON file" in str(cm.value)
        metadata_params = {
            "metadata_conventions": config_dict_general.get("metadata_conventions"),
            "metadata_path": "tests/example_files/metadata_ifdo_no_version.json",
            "metadata_type": config_dict_general["metadata_type"],
        }
        with pytest.raises(ValidationError) as cm:
            MetadataParser(**metadata_params)
        assert "No iFDO version found in metadata." in str(cm.value)


if __name__ == "__main__":
    unittest.main()
