""" Tests for the Simple Pipeline class.
"""

import glob
import os
import unittest
import numpy as np
import pandas as pd
from IPython.display import HTML
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass

class TestSimplePipeline(BaseTestClass):
    """Tests Simple Pipeline.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_simple_pipeline(self):
        """ Test generating a Simple Pipeline """
        pipeline = Pipeline(config_file_path="examples/config_files/config_benthic.yaml")
        self.assertTrue(isinstance(pipeline, Pipeline))
        self.assertTrue(isinstance(pipeline.config, Configuration))
        self.assertTrue(isinstance(pipeline.config.general, GeneralConfig))
        self.assertTrue(isinstance(pipeline.to_html(), str))
        pipeline.run()
        images = pipeline.images.images
        self.assertEqual(len(images), 7)
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        pipeline.run(from_step=2)
        images = pipeline.images.images
        self.assertEqual(len(images), 7)
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        metadata = pipeline.get_metadata()
        self.assertTrue(isinstance(metadata, pd.DataFrame))
        html_image = pipeline.images.show(image_number=5)
        self.assertTrue(isinstance(html_image, HTML))
        pipeline.save_images(image_format="png")
        output_files = glob.glob(
            os.path.join(pipeline.config.general.output_path, "*.png")
        )
        self.assertTrue(len(output_files) > 0)
        pipeline.images.remove()
        output_files = glob.glob(
            os.path.join(pipeline.config.general.output_path, "*.png")
        )
        self.assertTrue(len(output_files) == 0)

if __name__ == "__main__":
    unittest.main()
