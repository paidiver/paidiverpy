"""Tests for Pipeline Interaction."""

import unittest
from pathlib import Path
import numpy as np
from paidiverpy.colour_layer.colour_layer import ColourLayer
from paidiverpy.config.config import Configuration
from paidiverpy.config.config import GeneralConfig
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
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline.to_html(), str)
        pipeline.run()
        assert pipeline.steps[-1][2]["test"] is False
        assert len(pipeline.steps) == 3
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 3
        pipeline.add_step(
            "Area1",
            ResampleLayer,
            {"mode": "fixed", "params": {"value": 10}, "test": False},
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        assert len(pipeline.steps) == 3
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 3
        pipeline.add_step("contrast", ColourLayer, {"mode": "contrast"})
        assert len(pipeline.steps) == 4
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 4

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
        assert len(pipeline.steps) == 4
        pipeline.export_config("./new_config_pelagic.yaml")
        pipeline = Pipeline(config_file_path="./new_config_pelagic.yaml")
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == 4
        config_output_path = Path("./new_config_pelagic.yaml")
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == 1
        config_output_path.unlink()
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == 0


if __name__ == "__main__":
    unittest.main()
