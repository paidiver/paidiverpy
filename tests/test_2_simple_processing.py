""" Tests for the MetOfficeOcean class.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from paidiverpy.config.config import Configuration
from paidiverpy.open_layer import OpenLayer
from tests.base_test_class import BaseTestClass

class TestSimpleProcessing(BaseTestClass):
    """Tests Simple Processing.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_open_layer_class(self):
        open_layer = OpenLayer(config_file_path="tests/config_files/config_simple.yaml")
        self.assertTrue(isinstance(open_layer, OpenLayer))
        self.assertTrue(isinstance(open_layer.config, Configuration))
        open_layer.run()
        self.assertTrue(len(open_layer.images.images) > 0)
        self.assertTrue(isinstance(open_layer.images.images[0][0], np.ndarray))
        metadata = open_layer.get_metadata()
        self.assertTrue(isinstance(metadata, pd.DataFrame))


if __name__ == "__main__":
    unittest.main()
