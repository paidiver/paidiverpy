""" Tests for Pipeline with Custom Algorithm.
"""

import unittest
import pandas as pd
import numpy as np
from paidiverpy.config.config import Configuration, GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineCustomAlgorithm(BaseTestClass):
    """Tests for Pipeline with Custom Algorithm.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_custom_algorithm(self):
        """ Test generating a Pipeline with Custom Algorithm """
        pipeline = Pipeline(config_file_path="examples/config_files/config_custom_algorithm.yaml", verbose=1)

        self.assertTrue(isinstance(pipeline, Pipeline))
        self.assertTrue(isinstance(pipeline.config, Configuration))
        self.assertTrue(isinstance(pipeline.config.general, GeneralConfig))
        self.assertTrue(isinstance(pipeline.to_html(), str))
        self.assertTrue(isinstance(pipeline.get_metadata(), pd.DataFrame))
        self.assertEqual(pipeline.steps[-1][2]["step_name"], "custom")
        self.assertEqual(len(pipeline.steps), 3)
        pipeline.run()
        images = pipeline.images.images
        self.assertTrue(isinstance(images[0][0], np.ndarray))
        self.assertEqual(len(images), 3)

if __name__ == "__main__":
    unittest.main()
