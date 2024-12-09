""" Tests for the Config and Metadata class.
"""

from pathlib import Path
import unittest

from unittest.mock import patch
from datetime import datetime
import os
import glob
from paidiverpy.colour_layer.colour_layer import ColourLayer
from paidiverpy.convert_layer.convert_layer import ConvertLayer
from paidiverpy.custom_layer.custom_layer import CustomLayer
from paidiverpy.position_layer.position_layer import PositionLayer
import pandas as pd
from tests.base_test_class import BaseTestClass
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.open_layer import OpenLayer

class TestConfigMetadataClass(BaseTestClass):
    """Tests for the Config and Metadata class.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_config_class(self):
        """Test the Config class."""

        config = Configuration(config_file_path="tests/config_files/config_simple.yaml")
        config_dict = config.to_dict()
        self.assertTrue(isinstance(config_dict, dict))

    def test_parsing_config_file(self):
        """Test the parsing of the configuration file."""

        classes = [Paidiverpy, OpenLayer]
        for class_name in classes:
            paidiver = class_name(config_file_path="tests/config_files/config_simple.yaml")
            self.check_config(paidiver)

    def test_parsing_metadata(self):
        """Test the parsing of the configuration file."""

        config = Configuration(config_file_path="tests/config_files/config_simple.yaml")
        metadata = MetadataParser(config=config)
        self.assertTrue(isinstance(metadata, MetadataParser))

    def check_config(self, paidiver):
        self.assertTrue(isinstance(paidiver.config, Configuration))
        general = paidiver.config.general
        self.assertEqual(general.input_path, (Path.home() / ".paidiverpy_cache/benthic_csv/images").absolute())
        self.assertEqual(str(general.output_path), "output")
        self.assertTrue(len(general.sampling) > 0)
        steps = paidiver.config.steps
        self.assertTrue(len(steps) == 0)
        metadata = paidiver.get_metadata()
        self.assertTrue(isinstance(metadata, pd.DataFrame))


if __name__ == "__main__":
    unittest.main()
