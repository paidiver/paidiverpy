"""Tests for the Dask Cluster."""

import unittest
import pytest
from paidiverpy.utils.data import PaidiverpyData
from tests.base_test_class import BaseTestClass


class TestDataUtils(BaseTestClass):
    """Tests data utilities.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_data_utils_error_dataset(self):
        """Test generating a Pipeline with Custom Algorithm."""
        data = PaidiverpyData()
        with pytest.raises(ValueError) as cm:
            data.load("error")
        assert str(cm.value) == "Dataset 'error' not found."

    # def test_remove_dataset_and_download_again(self):
    #     """Test generating a Pipeline with Custom Algorithm."""
    #     data = PaidiverpyData()
    #     path_dir = Path.home() / ".paidiverpy_cache"
    #     if path_dir.exists():
    #         shutil.rmtree(path_dir)
    #     data.load("plankton_csv")
    #     data.load("plankton_csv")


if __name__ == "__main__":
    unittest.main()
