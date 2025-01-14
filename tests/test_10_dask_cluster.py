"""Tests for the Dask Cluster."""

import unittest
import numpy as np
import pandas as pd
from paidiverpy.config.config import Configuration
from paidiverpy.config.config import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestDaskCluster(BaseTestClass):
    """Tests for Pipeline using Dask Cluster.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_dask_cluster(self):
        """Test generating a Pipeline with Custom Algorithm."""
        pipeline = Pipeline(config_file_path="examples/config_files/config_benthic_client.yaml", verbose=2)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline.to_html(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert images[1][0] is None
        assert isinstance(images[-1][0], np.ndarray)
        assert len(images) == 7


if __name__ == "__main__":
    unittest.main()
