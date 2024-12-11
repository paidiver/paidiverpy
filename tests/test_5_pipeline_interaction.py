""" Tests for Pipeline Interaction.
"""

import glob
from pathlib import Path
import unittest
import numpy as np
from paidiverpy.colour_layer.colour_layer import ColourLayer
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from paidiverpy.resample_layer.resample_layer import ResampleLayer
from tests.base_test_class import BaseTestClass

class TestPipelineInteraction(BaseTestClass):
    """Tests for Pipeline Interaction.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_interaction(self):
        """Test the Pipeline Interaction."""
        pipeline = Pipeline(config_file_path="examples/config_files/config_pelagic.yaml")
        self.assertTrue(isinstance(pipeline, Pipeline))
        self.assertTrue(isinstance(pipeline.config, Configuration))
        self.assertTrue(isinstance(pipeline.config.general, GeneralConfig))
        self.assertTrue(isinstance(pipeline.to_html(), str))
        pipeline.run()
        self.assertTrue(pipeline.steps[-1][2]["test"] == False)
        self.assertEqual(len(pipeline.steps), 3)
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 3)
        pipeline.add_step(
            "Area1",
            ResampleLayer,
            {"mode": "fixed", "params": {"value": 10}, "test": False},
            1,
            substitute=True,
        )
        self.assertTrue(pipeline.steps[-1][2]["test"] == False)
        self.assertEqual(len(pipeline.steps), 3)
        pipeline.run(from_step=0)
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 3)
        pipeline.add_step("contrast", ColourLayer, {"mode": "contrast"})
        self.assertEqual(len(pipeline.steps), 4)
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 4)

    def test_pipeline_export(self):
        """Test the Pipeline Export."""
        pipeline = Pipeline(config_file_path="examples/config_files/config_pelagic.yaml")
        pipeline.add_step(
            "Area1",
            ResampleLayer,
            {"mode": "fixed", "params": {"value": 10}, "test": False},
            1,
            substitute=True,
        )
        pipeline.add_step("contrast", ColourLayer, {"mode": "contrast"})
        pipeline.run()
        self.assertEqual(len(pipeline.steps), 4)
        pipeline.export_config("./new_config_pelagic.yaml")
        pipeline = Pipeline(config_file_path="./new_config_pelagic.yaml")
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 4)
        config_output_path = Path("./new_config_pelagic.yaml")
        output_files = glob.glob(str(config_output_path.absolute()))
        self.assertTrue(len(output_files) == 1)
        config_output_path.unlink()
        output_files = glob.glob(str(config_output_path.absolute()))
        self.assertTrue(len(output_files) == 0)

if __name__ == "__main__":
    unittest.main()
