"""Tests for Pipeline with IFDO metadata."""

import unittest
from pathlib import Path
import dask.array as da
import pandas as pd
from IPython.display import HTML
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineIfdo(BaseTestClass):
    """Tests for Pipeline with IFDO metadata.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_ifdo(self):
        """Test generating a Pipeline with IFDO metadata."""
        number_images = 4
        number_output_files = 0

        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_ifdo.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], da.core.Array)
        assert len(images) == number_images
        html_image = pipeline.images.show(image_number=2)
        assert isinstance(html_image, HTML)
        output_path = Path(pipeline.config.general.output_path)
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) == number_output_files
        pipeline.save_images(image_format="png")
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) > number_output_files
        pipeline.images.remove()
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) == number_output_files


if __name__ == "__main__":
    unittest.main()
