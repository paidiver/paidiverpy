"""Tests for the Simple Pipeline class."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from IPython.display import HTML
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.convert_layer.convert_layer import ConvertLayer
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestSimplePipeline(BaseTestClass):
    """Tests Simple Pipeline.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_simple_pipeline(self):
        """Test generating a Simple Pipeline."""
        number_images = 7
        number_output_files = 0
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        pipeline.run(from_step=2)
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        metadata = pipeline.get_metadata()
        assert isinstance(metadata, pd.DataFrame)
        html_image = pipeline.images.show(image_number=5)
        assert isinstance(html_image, HTML)
        pipeline.save_images(image_format="png")
        output_path = Path(pipeline.config.general.output_path)
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) > number_output_files
        pipeline.images.remove()
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) == number_output_files
        pipeline.save_images(image_format="tiff", output_path=str(output_path.absolute()))
        output_files = list(output_path.glob("*.tiff"))
        assert len(output_files) > number_output_files
        pipeline.remove_images()
        output_files = list(output_path.glob("*.tiff"))
        assert len(output_files) == number_output_files
        output_repr = pipeline.images.__repr__()
        assert isinstance(output_repr, str)
        assert "Image:" in output_repr
        output_html = pipeline.images._repr_html_()
        assert isinstance(output_html, str)
        assert "<style>" in output_html
        assert "ppy-images-img" in output_html
        output_pipeline_str = pipeline._repr_html_()
        assert isinstance(output_pipeline_str, str)
        assert "parameters" in output_pipeline_str

    def test_simple_pipeline_uint16(self):
        """Test generating a Simple Pipeline."""
        number_images = 7
        number_output_files = 0
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_uint16.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        pipeline.run(from_step=2)
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        metadata = pipeline.get_metadata()
        assert isinstance(metadata, pd.DataFrame)
        html_image = pipeline.images.show(image_number=5)
        assert isinstance(html_image, HTML)
        pipeline.save_images(image_format="png")
        output_path = Path(pipeline.config.general.output_path)
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) > number_output_files
        pipeline.images.remove()
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) == number_output_files
        pipeline.save_images(image_format="tiff", output_path=str(output_path.absolute()))
        output_files = list(output_path.glob("*.tiff"))
        assert len(output_files) > number_output_files
        pipeline.images.remove()
        output_files = list(output_path.glob("*.tiff"))
        assert len(output_files) == number_output_files
        pipeline.add_step(
            "normalize",
            ConvertLayer,
            {"mode": "normalize", "test": False},
        )
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images + 1
        output_str = pipeline.images._repr_html_()
        assert isinstance(output_str, str)
        output_html = pipeline.images.__call__()
        assert isinstance(output_html, HTML)

    def test_force_error_verbose(self):
        """Test force error verbose."""
        with pytest.raises(ValueError) as cm:
            Pipeline(config_file_path="tests/config_files/config_benthic.yml", verbose=20)
        assert "Invalid verbose level:" in str(cm.value)


if __name__ == "__main__":
    unittest.main()
