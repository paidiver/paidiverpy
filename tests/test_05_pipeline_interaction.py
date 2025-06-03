"""Tests for Pipeline Interaction."""

import unittest
from pathlib import Path
import numpy as np
import pytest
from paidiverpy.colour_layer.colour_layer import ColourLayer
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from paidiverpy.sampling_layer.sampling_layer import SamplingLayer
from tests.base_test_class import BaseTestClass


class TestPipelineInteraction(BaseTestClass):
    """Tests for Pipeline Interaction.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_interaction(self):
        """Test the Pipeline Interaction."""
        number_images = 3
        number_pipeline_steps = 3

        pipeline = Pipeline(config_file_path="tests/config_files/config_plankton.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        assert pipeline.steps[-1][2]["test"] is False
        assert len(pipeline.steps) == number_pipeline_steps
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images
        pipeline.add_step(
            "Area1",
            SamplingLayer,
            {"mode": "fixed", "params": {"value": 10}, "test": False},
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        assert len(pipeline.steps) == number_pipeline_steps
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images
        pipeline.add_step("contrast", ColourLayer, {"mode": "contrast"})
        assert len(pipeline.steps) == number_pipeline_steps + 1
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images + 1

    def test_pipeline_export(self):
        """Test the Pipeline Export."""
        number_images = 4
        number_pipeline_steps = 4
        number_output_files = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_plankton.yml")
        pipeline.add_step(
            "Area1",
            SamplingLayer,
            {"mode": "fixed", "params": {"value": 10}, "test": False},
            1,
            substitute=True,
        )
        pipeline.add_step("contrast", ColourLayer, {"mode": "contrast"})
        pipeline.run()
        assert len(pipeline.steps) == number_pipeline_steps
        pipeline.export_config("./new_config_plankton.yml")
        pipeline = Pipeline(config_file_path="./new_config_plankton.yml")
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images
        config_output_path = Path("./new_config_plankton.yml")
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == number_output_files
        config_output_path.unlink()
        output_files = list(config_output_path.parent.glob(config_output_path.name))
        assert len(output_files) == 0

    def test_raise_error(self):
        """Test the Pipeline Raise Error."""
        pipeline = Pipeline(config_file_path="tests/config_files/config_plankton.yml")
        pipeline.add_step(
            "sharpen_error",
            ColourLayer,
            {"mode": "sharpen", "params": {"alpha": -10, "beta": 0.5, "raise_error": True}, "test": False},
            1,
            substitute=True,
        )
        with pytest.raises(ValueError) as cm:
            pipeline.run()
        assert "Error applying sharpening:" in str(cm.value)


if __name__ == "__main__":
    unittest.main()
