"""Tests for the Simple Pipeline class."""

import unittest
from pathlib import Path
import numpy as np
import pytest
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass

number_graphs = 1


class TestPositionLayer(BaseTestClass):
    """Tests Simple Pipeline.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_position_layer(self):
        """Test the position layer."""
        number_images = 2
        pipeline = Pipeline(config_file_path="tests/config_files/config_position.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_calculate_corners/*graph_*.png"))
        assert len(output_graphs) == number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()

    def test_position_layer_error(self):
        """Test the position layer."""
        pipeline = Pipeline(config_file_path="tests/config_files/config_position_error.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        with pytest.raises(ValueError) as cm:
            pipeline.run()
        assert str(cm.value) == "Position layer step failed."


if __name__ == "__main__":
    unittest.main()
