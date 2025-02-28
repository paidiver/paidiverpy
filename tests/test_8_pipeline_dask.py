"""Tests for Pipeline with Parallel Processing using dask."""

import unittest
import dask.array as da
import pandas as pd
from IPython.display import HTML
from paidiverpy.config.config import Configuration
from paidiverpy.config.config import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineDask(BaseTestClass):
    """Tests for Pipeline with Parallel Processing using dask.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_parallel_processing_dask(self):
        """Test generating a Pipeline with Parallel Processing using dask."""
        number_images = 7

        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_dask.yml", verbose=0)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline.to_html(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], da.core.Array)
        assert len(images) == number_images
        html_image = pipeline.images.show(image_number=2)
        assert isinstance(html_image, HTML)


if __name__ == "__main__":
    unittest.main()
