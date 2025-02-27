"""Tests for Pipeline with Custom Algorithm."""

import unittest
import numpy as np
import pandas as pd
from paidiverpy.config.config import Configuration
from paidiverpy.config.config import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineCustomAlgorithm(BaseTestClass):
    """Tests for Pipeline with Custom Algorithm.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_custom_algorithm(self):
        """Test generating a Pipeline with Custom Algorithm."""
        number_pipeline_steps = 3
        number_images = 3

        pipeline = Pipeline(config_file_path="tests/config_files/config_custom_algorithm.yml", verbose=1)

        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline.to_html(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        assert pipeline.steps[-1][2]["step_name"] == "custom"
        assert len(pipeline.steps) == number_pipeline_steps
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images


if __name__ == "__main__":
    unittest.main()
