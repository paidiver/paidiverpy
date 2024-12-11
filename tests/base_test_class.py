""" Tests for the Paidiverpy package class.
"""

from pathlib import Path
import shutil
import unittest

import logging

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*jsonschema.RefResolver is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*distutils Version classes are deprecated.*"
)

class BaseTestClass(unittest.TestCase):
    """Base test class for the paidiverpy package"""

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(cls.__name__)
        cls.remove_datasets()
        cls.remove_custom_packages()

    @classmethod
    def tearDownClass(cls):
        cls.remove_datasets()
        cls.remove_custom_packages()

    @classmethod
    def remove_datasets(cls):
        """Remove the datasets."""
        path_dir = Path.home() / ".paidiverpy_cache"
        if path_dir.exists():
            try:
                shutil.rmtree(path_dir)
                cls.logger.info(f"Removed cache directory: {path_dir}")
            except Exception as e:
                cls.logger.error(f"Error removing cache directory: {e}")

    @classmethod
    def remove_custom_packages(cls):
        """Remove the custom packages."""
        path_dir = Path.cwd() / "custom_packages"
        if path_dir.exists():
            try:
                shutil.rmtree(path_dir)
                cls.logger.info(f"Removed custom packages directory: {path_dir}")
            except Exception as e:
                cls.logger.error(f"Error removing custom packages directory: {e}")

if __name__ == "__main__":
    unittest.main()
