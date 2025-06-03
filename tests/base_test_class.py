"""Tests for the Paidiverpy package class."""

import shutil
import unittest
import warnings
from pathlib import Path
from paidiverpy.utils.logging_functions import initialise_logging

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*jsonschema.RefResolver is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*distutils Version classes are deprecated.*")

NEF_RAW_LINK = "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/nef_raw_examples/"
RAW_IMAGES_LINK = "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/raw_data_examples/"


class BaseTestClass(unittest.TestCase):
    """Base test class for the paidiverpy package."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test class."""
        cls.logger = initialise_logging()
        cls.cleanup_directories()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down the test class."""
        cls.cleanup_directories()

    @classmethod
    def cleanup_directories(cls) -> None:
        """Cleanup the directories."""
        path_dirs = [
            Path("output"),
            # Path.home() / ".paidiverpy_cache",
            Path.cwd() / "custom_packages",
        ]
        for path_dir in path_dirs:
            if path_dir.exists():
                try:
                    shutil.rmtree(path_dir)
                    cls.logger.info("Removed directory: %s", path_dir)
                except FileNotFoundError:
                    cls.logger.warning("Directory not found: %s", path_dir)
                except PermissionError:
                    cls.logger.error("Permission denied while removing: %s", path_dir)
                except OSError as e:
                    cls.logger.error("OS error while removing directory %s: %s", path_dir, e)


if __name__ == "__main__":
    unittest.main()
