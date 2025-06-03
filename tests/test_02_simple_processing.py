"""Tests for the Simple Processing without creating pipeline."""

import unittest
import numpy as np
import pandas as pd
import pytest
from IPython.display import HTML
from paidiverpy.config.configuration import Configuration
from paidiverpy.open_layer import OpenLayer
from paidiverpy.sampling_layer.sampling_layer import SamplingLayer
from paidiverpy.utils.data import PaidiverpyData
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
            "params": {"min": "2018-06-11 10:18:00", "max": "2018-06-11 04:20:00"},
        }
        resample_layer = SamplingLayer(paidiverpy=open_layer, parameters=parameters)
        resample_layer.run()
        images = resample_layer.images.images
        assert len(images) == number_images
        parameters = {
            "name": "datetime",
            "mode": "datetime",
            "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
        }
        resample_layer = SamplingLayer(paidiverpy=open_layer, parameters=parameters)
        assert isinstance(resample_layer, SamplingLayer)
        resample_layer_config = resample_layer.config
        assert isinstance(resample_layer_config, Configuration)
        assert resample_layer_config.general == open_layer_config.general
        assert resample_layer_config.steps == open_layer_config.steps
        resample_layer.run()
        images = resample_layer.images.images
        assert len(images) == number_images + 1
        parameters = {
            "name": "datetime",
            "params": {"min": "2018-06-11 04:18:00", "max": "2018-06-11 04:20:00"},
        }
        with pytest.raises(ValueError) as cm:
            resample_layer = SamplingLayer(paidiverpy=open_layer, parameters=parameters)
        assert str(cm.value) == "Mode is not defined for the resample layer."
        parameters = {
            "name": "datetime",
            "mode": "datetime",
            "params": {"min": "2018-06-11 10:18:00", "max": "2018-06-11 04:20:00"},
        }
        open_layer.raise_error = True
        with pytest.raises(ValueError) as cm:
            SamplingLayer(paidiverpy=open_layer, parameters=parameters).run()
        assert str(cm.value) == "Sampling layer step failed."

    def test_processing_without_conf_file(self):
        """Test the OpenLayer class."""
        number_images = 2
        data = PaidiverpyData()
        config_params = data.load("plankton_csv")
        with pytest.raises(ValueError) as cm:
            OpenLayer(config_params=config_params)
        assert "Error in config_params: params ['output_path']" in str(cm.value)
        config_params["output_path"] = "tests/output"
        open_layer = OpenLayer(config_params=config_params)
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
            "params": {"min": "2018-06-11 10:18:00", "max": "2018-06-11 04:20:00"},
        }
        resample_layer = SamplingLayer(paidiverpy=open_layer, parameters=parameters)
        resample_layer.run()
        images = resample_layer.images.images
        assert len(images) == number_images
        parameters = {
            "name": "datetime",
            "mode": "datetime",
            "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
        }
        resample_layer = SamplingLayer(paidiverpy=open_layer, parameters=parameters)
        assert isinstance(resample_layer, SamplingLayer)
        resample_layer_config = resample_layer.config
        assert isinstance(resample_layer_config, Configuration)
        assert resample_layer_config.general == open_layer_config.general
        assert resample_layer_config.steps == open_layer_config.steps
        resample_layer.run()
        images = resample_layer.images.images
        assert len(images) == number_images + 1


if __name__ == "__main__":
    unittest.main()
