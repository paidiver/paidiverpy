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
            config_file_path="examples/config_files/config_simple.yaml",
            steps=pipeline_steps,
        )
        assert len(pipeline.steps) == 1
        assert len(pipeline.config.general.sampling) == 1
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 1

    def test_pipeline_generator2(self):
        """Test generating a Pipeline 2."""
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
            config_file_path="examples/config_files/config_simple.yaml",
            steps=pipeline_steps,
        )
        assert len(pipeline.steps) == 2
        assert len(pipeline.config.general.sampling) == 1
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 2
        pipeline.export_config("new_config.yaml")
        config_output_path = Path("./new_config.yaml")
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == 1
        pipeline = Pipeline(config_file_path="new_config.yaml")
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 2
        config_output_path.unlink()
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == 0


if __name__ == "__main__":
    unittest.main()
