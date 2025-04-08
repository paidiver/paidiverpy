"""Tests for the Simple Pipeline class."""

import unittest
from tests.base_test_class import BaseTestClass

number_graphs = 1


class TestTrackChanges(BaseTestClass):
    """Tests Track Changes.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_processing_remote(self):
        """Test no track changes."""
        assert True


if __name__ == "__main__":
    unittest.main()
