"""Tests for the Simple Pipeline class."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import HTML
from paidiverpy.config.config import Configuration
from paidiverpy.config.config import GeneralConfig
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
        assert isinstance(pipeline.to_html(), str)
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


if __name__ == "__main__":
    unittest.main()
