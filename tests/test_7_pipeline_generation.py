"""Tests for Pipeline Generation."""

import unittest
from pathlib import Path
import numpy as np
from paidiverpy.colour_layer import ColourLayer
from paidiverpy.open_layer import OpenLayer
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineGenerator(BaseTestClass):
    """Tests for Pipeline Generation.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_generator1(self):
        """Test generating a Pipeline 1."""
        number_images = 1
        number_pipeline_steps = 1
        number_general_sampling = 1

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
            config_file_path="tests/config_files/config_simple.yml",
            steps=pipeline_steps,
        )
        assert len(pipeline.steps) == number_pipeline_steps
        assert len(pipeline.config.general.sampling) == number_general_sampling
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images

    def test_pipeline_generator2(self):
        """Test generating a Pipeline 2."""
        number_images = 2
        number_pipeline_steps = 2
        number_general_sampling = 1
        number_output_files = 1

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
            config_file_path="tests/config_files/config_simple.yml",
            steps=pipeline_steps,
        )
        assert len(pipeline.steps) == number_pipeline_steps
        assert len(pipeline.config.general.sampling) == number_general_sampling
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images
        pipeline.export_config("new_config.yml")
        config_output_path = Path("./new_config.yml")
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == number_output_files
        pipeline = Pipeline(config_file_path="new_config.yml")
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images
        config_output_path.unlink()
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == 0


if __name__ == "__main__":
    unittest.main()
