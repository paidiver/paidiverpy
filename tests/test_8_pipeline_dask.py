""" Tests for Pipeline with Parallel Processing using dask
"""

import unittest
import pandas as pd
import dask.array as da
from IPython.display import HTML
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass

import warnings

class TestPipelineDask(BaseTestClass):
    """Tests for Pipeline with Parallel Processing using dask.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_parallel_processing_dask(self):
        """ Test generating a Pipeline with Parallel Processing using dask """

        pipeline = Pipeline(config_file_path="examples/config_files/config_benthic_dask.yaml", verbose=0)
        self.assertTrue(isinstance(pipeline, Pipeline))
        self.assertTrue(isinstance(pipeline.config, Configuration))
        self.assertTrue(isinstance(pipeline.config.general, GeneralConfig))
        self.assertTrue(isinstance(pipeline.to_html(), str))
        self.assertTrue(isinstance(pipeline.get_metadata(), pd.DataFrame))
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], da.core.Array))
        self.assertEqual(len(images), 7)
        html_image = pipeline.images.show(image_number=2)
        self.assertTrue(isinstance(html_image, HTML))

if __name__ == "__main__":
    unittest.main()
