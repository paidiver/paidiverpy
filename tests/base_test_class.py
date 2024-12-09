""" Tests for the Paidiverpy package class.
"""

import unittest

import logging
import os
import subprocess
import zipfile
from dotenv import load_dotenv

load_dotenv(override=True)


class BaseTestClass(unittest.TestCase):
    """Base test class for the paidiverpy package"""

    @classmethod
    def setUpClass(cls):
        # if os.environ.get("ROOT_PATH"):
        #     cls.root_dir = os.path.join(os.environ.get("ROOT_PATH"), "tests")
        # else:
        #     cls.root_dir = os.path.dirname(__file__)
        # cls.test_dir = os.path.join(cls.root_dir, "test_data")
        # cls.local_path = os.path.join(cls.root_dir, "sample_data")
        # if not os.path.exists(cls.test_dir):
        #     os.mkdir(cls.test_dir)
        # if not os.path.exists(cls.local_path):
        #     os.mkdir(cls.local_path)

        cls.logger = logging.getLogger(cls.__name__)
        # cls.zip_file_path = os.path.join(cls.root_dir, "sample_data.zip")

        # cls.download_and_extract_files()

    @classmethod
    def tearDownClass(cls):
        pass
        # for item in os.listdir(cls.test_dir):
        #     item_path = os.path.join(cls.test_dir, item)
        #     if os.path.isfile(item_path):
        #         os.unlink(item_path)
        #     elif os.path.isdir(item_path):
        #         os.rmdir(item_path)


if __name__ == "__main__":
    unittest.main()
