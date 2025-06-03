"""Tests for Raw Files."""

import unittest
import numpy as np
from paidiverpy.config.configuration import Configuration
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import NEF_RAW_LINK
from tests.base_test_class import RAW_IMAGES_LINK
from tests.base_test_class import BaseTestClass

number_graphs = 1


class TestRawFiles(BaseTestClass):
    """Tests Raw Files.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_raw_nef_files(self):
        """Test raw nef files."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/raw_images/config_raw_images_nef.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)

    def test_raw_nef_files_remote(self):
        """Test raw nef files."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/raw_images/config_raw_images_nef.yml")
        pipeline.config.general.input_path = NEF_RAW_LINK
        pipeline.steps[0][2]["input_path"] = NEF_RAW_LINK
        pipeline.config.general.sample_data = None
        pipeline.steps[0][2]["sample_data"] = None
        pipeline.config.is_remote = True
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)

    def test_raw_files(self):
        """Test raw nef files."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/raw_images/config_raw_images.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)

    def test_raw_files_remote(self):
        """Test raw nef files."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/raw_images/config_raw_images.yml")
        pipeline.config.general.input_path = RAW_IMAGES_LINK
        pipeline.steps[0][2]["input_path"] = RAW_IMAGES_LINK
        pipeline.config.general.sample_data = None
        pipeline.steps[0][2]["sample_data"] = None
        pipeline.config.is_remote = True
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)


if __name__ == "__main__":
    unittest.main()
