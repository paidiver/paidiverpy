""" Tests for the MetOfficeOcean class.
"""

import unittest

import logging
import os
import subprocess
import zipfile
from dotenv import load_dotenv

load_dotenv()


class BaseTestClass(unittest.TestCase):
    """Base test class for the met_office_data_handler package"""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("ROOT_PATH"):
            cls.root_dir = os.path.join(os.environ.get("ROOT_PATH"), "tests")
        else:
            cls.root_dir = os.path.dirname(__file__)
        cls.test_dir = os.path.join(cls.root_dir, "test_data")
        cls.local_path = os.path.join(cls.root_dir, "sample_data")
        if not os.path.exists(cls.test_dir):
            os.mkdir(cls.test_dir)
        if not os.path.exists(cls.local_path):
            os.mkdir(cls.local_path)

        cls.logger = logging.getLogger(cls.__name__)
        cls.zip_file_path = os.path.join(cls.root_dir, "sample_data.zip")

        cls.download_and_extract_files()

    @classmethod
    def tearDownClass(cls):
        for item in os.listdir(cls.test_dir):
            item_path = os.path.join(cls.test_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                os.rmdir(item_path)

    @classmethod
    def download_and_extract_files(cls):
        """Download and extract test files if not already done."""
        if not os.path.isfile(cls.zip_file_path):
            url = os.environ.get("SAMPLE_DATA_URL")
            subprocess.run(["wget", url, "-O", cls.zip_file_path], check=True)

        # Extract the zip file if contents are not already extracted
        if not os.listdir(cls.local_path):
            with zipfile.ZipFile(cls.zip_file_path, "r") as zip_ref:
                zip_ref.extractall(cls.root_dir)


if __name__ == "__main__":
    unittest.main()
