"""Tests for the Dask Cluster."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestDaskCluster(BaseTestClass):
    """Tests for Pipeline using Dask Cluster.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_dask_cluster(self):
        """Test generating a Pipeline with Custom Algorithm."""
        number_output_files = 0
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_client.yml", verbose=2)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        pipeline.run(close_client=False)
        images = pipeline.images.images
        assert "images_0" not in images.data_vars
        assert "images_1" not in images.data_vars
        assert "images_2" not in images.data_vars
        assert "images_3" not in images.data_vars
        assert isinstance(images["images_4"].values[0], np.ndarray)
        output_path = Path(pipeline.config.general.output_path)
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) == number_output_files
        pipeline.save_images(image_format="png")
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) > number_output_files
        pipeline.images.remove()
        output_files = list(output_path.glob("*.png"))
        assert len(output_files) == number_output_files
        pipeline.client.close()


if __name__ == "__main__":
    unittest.main()
