""" Tests for Pipeline Generation.
"""

import glob
from pathlib import Path
import unittest
import numpy as np
from paidiverpy.pipeline import Pipeline
from paidiverpy.colour_layer import ColourLayer
from paidiverpy.open_layer import OpenLayer
from tests.base_test_class import BaseTestClass


class TestPipelineGenerator(BaseTestClass):
    """Tests for Pipeline Generation.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_generator1(self):
        """ Test generating a Pipeline 1 """

        open_layer_params = {
            "convert": [
                {
                    "mode": "bits",
                    "params": {
                        "output_bits": 8,
                    },
                },
            ],
            "sampling": [{"mode": "percent", "params": {"value": 0.3}}],
        }
        pipeline_steps = [("raw", OpenLayer, open_layer_params)]
        pipeline = Pipeline(
            config_file_path="examples/config_files/config_simple.yaml", steps=pipeline_steps,
        )
        self.assertEqual(len(pipeline.steps), 1)
        self.assertEqual(len(pipeline.config.general.sampling), 1)
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 1)

    def test_pipeline_generator2(self):
        """ Test generating a Pipeline 2 """

        open_layer_params = {
            "convert": [
                {
                    "mode": "bits",
                    "params": {
                        "output_bits": 8,
                    },
                },
            ],
            "sampling": [{"mode": "percent", "params": {"value": 0.1}}],
        }
        pipeline_steps = [
            ("raw", OpenLayer, open_layer_params),
            ("gray", ColourLayer, {"mode": "grayscale"}),
        ]
        pipeline = Pipeline(
            config_file_path="examples/config_files/config_simple.yaml", steps=pipeline_steps,
        )
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(len(pipeline.config.general.sampling), 1)
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 2)
        pipeline.export_config("new_config.yaml")
        config_output_path = Path("./new_config.yaml")
        output_files = glob.glob(str(config_output_path.absolute()))
        self.assertTrue(len(output_files) == 1)
        pipeline = Pipeline(config_file_path="new_config.yaml")
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 2)
        config_output_path.unlink()
        output_files = glob.glob(str(config_output_path.absolute()))
        self.assertTrue(len(output_files) == 0)

if __name__ == "__main__":
    unittest.main()
