""" Tests for Pipeline Testing Steps.
"""

import unittest
from unittest.mock import patch
import numpy as np
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from paidiverpy.resample_layer.resample_layer import ResampleLayer
from tests.base_test_class import BaseTestClass

class TestPipelineTestSteps(BaseTestClass):
    """Tests for Pipeline Testing Steps.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    @patch('matplotlib.pyplot.show')
    def test_pipeline_testing_steps(self, mock_show):
        """Test the Pipeline Testing Steps."""
        pipeline = Pipeline(config_file_path="examples/config_files/config_benthic_test_steps.yaml")
        self.assertTrue(isinstance(pipeline, Pipeline))
        self.assertTrue(isinstance(pipeline.config, Configuration))
        self.assertTrue(isinstance(pipeline.config.general, GeneralConfig))
        self.assertTrue(isinstance(pipeline.to_html(), str))
        pipeline.run()
        self.assertEqual(mock_show.call_count, 2)
        images = pipeline.images.images
        self.assertEqual(len(images), 1)
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertTrue(pipeline.steps[1][2]["test"])
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
        self.assertTrue(pipeline.steps[-1][2]["test"] == False)
        pipeline.run(from_step=0)
        images = pipeline.images.images
        self.assertEqual(len(images), 2)
        self.assertEqual(mock_show.call_count, 2)
        pipeline.add_step(
            "datetime",
            ResampleLayer,
            {
                "mode": "datetime",
                "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
                "test": True,
            },
        )
        self.assertTrue(pipeline.steps[-1][2]["test"] == True)
        self.assertTrue(pipeline.steps[-1][0] == "datetime")
        pipeline.run()
        self.assertEqual(mock_show.call_count, 3)
        images = pipeline.images.images
        self.assertEqual(len(images), 2)
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
        self.assertTrue(pipeline.steps[-1][2]["test"] == False)
        self.assertTrue(pipeline.steps[-1][0] == "datetime")
        pipeline.run()
        self.assertEqual(mock_show.call_count, 3)
        images = pipeline.images.images
        self.assertEqual(len(images), 3)

if __name__ == "__main__":
    unittest.main()
