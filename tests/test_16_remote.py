"""Tests for Remote Data."""

import unittest
from tests.base_test_class import BaseTestClass

number_graphs = 1


class TestRemoteData(BaseTestClass):
    """Tests Remote Data.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_processing_remote(self):
        """Test no track changes."""
        assert True


if __name__ == "__main__":
    unittest.main()
