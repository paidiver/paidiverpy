"""Tests for Pipeline Testing Steps."""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
from paidiverpy.config.config import Configuration
from paidiverpy.config.config import GeneralConfig
from paidiverpy.pipeline import Pipeline
from paidiverpy.resample_layer.resample_layer import ResampleLayer
from tests.base_test_class import BaseTestClass


class TestPipelineTestSteps(BaseTestClass):
    """Tests for Pipeline Testing Steps.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    @patch("matplotlib.pyplot.show")
    def test_pipeline_testing_steps(self, mock_show: MagicMock):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        number_calls = 2
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline.to_html(), str)
        pipeline.run()
        assert mock_show.call_count == number_calls
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "overlapping",
            ResampleLayer,
            {
                "mode": "overlapping",
                "params": {"theta": 40, "omega": 57, "threshold": 0.1},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1
        assert mock_show.call_count == number_calls
        pipeline.add_step(
            "datetime",
            ResampleLayer,
            {
                "mode": "datetime",
                "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
                "test": True,
            },
        )
        assert pipeline.steps[-1][2]["test"] is True
        assert pipeline.steps[-1][0] == "datetime"
        pipeline.run()
        assert mock_show.call_count == number_calls + 1
        images = pipeline.images.images
        assert len(images) == number_images + 1
        pipeline.add_step(
            "datetime",
            ResampleLayer,
            {
                "mode": "datetime",
                "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
                "test": False,
            },
            2,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        assert pipeline.steps[-1][0] == "datetime"
        pipeline.run()
        assert mock_show.call_count == number_calls + 1
        images = pipeline.images.images
        assert len(images) == number_images + 2


if __name__ == "__main__":
    unittest.main()
