""" Tests for the Dask Cluster.
"""

import unittest
import pandas as pd
import numpy as np
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestDaskCluster(BaseTestClass):
    """Tests for Pipeline using Dask Cluster.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_dask_cluster(self):
        """ Test generating a Pipeline with Custom Algorithm """
        pipeline = Pipeline(config_file_path="examples/config_files/config_benthic_client.yaml", verbose=2)
        self.assertTrue(isinstance(pipeline, Pipeline))
        self.assertTrue(isinstance(pipeline.config, Configuration))
        self.assertTrue(isinstance(pipeline.config.general, GeneralConfig))
        self.assertTrue(isinstance(pipeline.to_html(), str))
        self.assertTrue(isinstance(pipeline.get_metadata(), pd.DataFrame))
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(images[1][0], None)
        self.assertTrue(isinstance(images[-1][0], np.ndarray))
        self.assertEqual(len(images), 7)

if __name__ == "__main__":
    unittest.main()
