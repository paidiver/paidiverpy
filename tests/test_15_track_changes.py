"""Tests for the Simple Pipeline class."""

import unittest
import numpy as np
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass

number_graphs = 1


class TestTrackChanges(BaseTestClass):
    """Tests Track Changes.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_processing_no_track_changes(self):
        """Test no track changes."""
        pipeline = Pipeline(config_file_path="tests/config_files/config_simple_no_track_changes.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert "images_0" not in images.data_vars
        assert "images_1" not in images.data_vars
        assert "images_2" not in images.data_vars
        assert "images_3" not in images.data_vars
        assert "images_4" not in images.data_vars
        assert "images_5" in images.data_vars
        assert isinstance(images["images_5"][0].values, np.ndarray)
        output_html = pipeline.images._repr_html_()
        assert isinstance(output_html, str)
        assert "Images for this step are not available" in output_html


if __name__ == "__main__":
    unittest.main()
