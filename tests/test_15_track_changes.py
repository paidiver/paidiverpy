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
        number_images = 6
        pipeline = Pipeline(config_file_path="tests/config_files/config_simple_no_track_changes.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert images[0][0] is None
        assert images[1][0] is None
        assert images[2][0] is None
        assert images[3][0] is None
        assert images[4][0] is None
        assert isinstance(images[5][0], np.ndarray)
        output_html = pipeline.images._repr_html_()
        assert isinstance(output_html, str)
        assert "No image to show" in output_html


if __name__ == "__main__":
    unittest.main()
