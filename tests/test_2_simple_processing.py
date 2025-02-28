"""Tests for the Simple Processing without creating pipeline."""

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
        """Test the OpenLayer class."""
        number_images = 2

        open_layer = OpenLayer(config_file_path="tests/config_files/config_simple.yml")
        assert isinstance(open_layer, OpenLayer)
        open_layer_config = open_layer.config
        assert isinstance(open_layer_config, Configuration)
        open_layer.run()
        assert len(open_layer.images.images) > 0
        assert isinstance(open_layer.images.images[0][0], np.ndarray)
        images = open_layer.images.images
        assert len(images) == 1
        metadata = open_layer.get_metadata()
        assert isinstance(metadata, pd.DataFrame)
        html_image = open_layer.images.show(image_number=5)
        assert isinstance(html_image, HTML)
        parameters = {
            "name": "datetime",
            "mode": "datetime",
            "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
        }
        resample_layer = ResampleLayer(paidiverpy=open_layer, parameters=parameters)
        assert isinstance(resample_layer, ResampleLayer)
        resample_layer_config = resample_layer.config
        assert isinstance(resample_layer_config, Configuration)
        assert resample_layer_config.general == open_layer_config.general
        assert resample_layer_config.steps == open_layer_config.steps
        resample_layer.run()
        images = resample_layer.images.images
        assert len(images) == number_images


if __name__ == "__main__":
    unittest.main()
