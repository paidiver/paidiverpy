"""Tests for the Paidiverpy package class."""

import logging
import shutil
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*jsonschema.RefResolver is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*distutils Version classes are deprecated.*")


class BaseTestClass(unittest.TestCase):
    """Base test class for the paidiverpy package."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test class."""
        cls.logger = logging.getLogger("paidiverpy")
        cls.remove_datasets()
        cls.remove_custom_packages()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down the test class."""
        cls.remove_datasets()
        cls.remove_custom_packages()

    @classmethod
    def remove_datasets(cls) -> None:
        """Remove the datasets."""
        path_dir = Path.home() / ".paidiverpy_cache"
        if path_dir.exists():
            try:
                shutil.rmtree(path_dir)
                cls.logger.info("Removed cache directory: %s", path_dir)
            except FileNotFoundError:
                cls.logger.warning("Directory not found: %s", path_dir)
            except PermissionError:
                cls.logger.error("Permission denied while removing: %s", path_dir)
            except OSError as e:
                cls.logger.error("OS error while removing directory %s: %s", path_dir, e)

    @classmethod
    def remove_custom_packages(cls) -> None:
        """Remove the custom packages."""
        path_dir = Path.cwd() / "custom_packages"
        if path_dir.exists():
            try:
                shutil.rmtree(path_dir)
                cls.logger.info("Removed custom packages directory: %s", path_dir)
            except FileNotFoundError:
                cls.logger.warning("Directory not found: %s", path_dir)
            except PermissionError:
                cls.logger.error("Permission denied while removing: %s", path_dir)
            except OSError as e:
                cls.logger.error("OS error while removing directory %s: %s", path_dir, e)


if __name__ == "__main__":
    unittest.main()
