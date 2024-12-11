""" Tests for the Simple Processing without creating pipeline.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from IPython.display import HTML
from paidiverpy.config.config import Configuration
from paidiverpy.open_layer import OpenLayer
from paidiverpy.resample_layer.resample_layer import ResampleLayer
from tests.base_test_class import BaseTestClass

class TestSimpleProcessing(BaseTestClass):
    """Tests Simple Processing.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_open_layer_class(self):
        """ Test the OpenLayer class """
        open_layer = OpenLayer(config_file_path="examples/config_files/config_simple.yaml")
        self.assertTrue(isinstance(open_layer, OpenLayer))
        open_layer_config = open_layer.config
        self.assertTrue(isinstance(open_layer_config, Configuration))
        open_layer.run()
        self.assertTrue(len(open_layer.images.images) > 0)
        self.assertTrue(isinstance(open_layer.images.images[0][0], np.ndarray))
        images = open_layer.images.images
        self.assertTrue(len(images) == 1)
        metadata = open_layer.get_metadata()
        self.assertTrue(isinstance(metadata, pd.DataFrame))
        html_image = open_layer.images.show(image_number=5)
        self.assertTrue(isinstance(html_image, HTML))
        parameters = {
            "name": "datetime",
            "mode": "datetime",
            "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
        }
        resample_layer = ResampleLayer(paidiverpy=open_layer, parameters=parameters)
        self.assertTrue(isinstance(resample_layer, ResampleLayer))
        resample_layer_config = resample_layer.config
        self.assertTrue(isinstance(resample_layer_config, Configuration))
        self.assertTrue(resample_layer_config.general == open_layer_config.general)
        print('1', resample_layer_config.steps)
        print('2', open_layer_config.steps)
        self.assertTrue(resample_layer_config.steps == open_layer_config.steps)
        resample_layer.run()
        images = resample_layer.images.images
        self.assertTrue(len(images) == 2)

if __name__ == "__main__":
    unittest.main()
