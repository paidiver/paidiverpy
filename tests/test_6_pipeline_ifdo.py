""" Tests for Pipeline with IFDO metadata.
"""

import glob
import os
import unittest
import pandas as pd
import dask.array as da
from IPython.display import HTML
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineIfdo(BaseTestClass):
    """Tests for Pipeline with IFDO metadata.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_ifdo(self):
        """ Test generating a Pipeline with IFDO metadata """
        pipeline = Pipeline(config_file_path="examples/config_files/config_benthic_ifdo.yaml")
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

        output_files = glob.glob(
            os.path.join(pipeline.config.general.output_path, "*.png")
        )
        self.assertTrue(len(output_files) == 0)

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
