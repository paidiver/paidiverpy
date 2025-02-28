"""Tests for the Config and Metadata class."""

import unittest
from pathlib import Path
import pandas as pd
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.open_layer import OpenLayer
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


if __name__ == "__main__":
    unittest.main()
