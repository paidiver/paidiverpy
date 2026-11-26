"""Tests for the generic OpenCV and scikit-image layer."""

import cv2
import numpy as np
import pytest
from paidiverpy.library_layer import LibraryLayer
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass

EXPECTED_PIPELINE_STEPS = 3


class TestLibraryLayer(BaseTestClass):
    """Test library callable resolution and pipeline processing."""

    def test_opencv_and_skimage_pipeline(self):
        """Apply functions from both supported libraries to every image."""
        pipeline = Pipeline(config_file_path="examples/config_files/config_library.yml", verbose=1)
        pipeline.run()

        assert pipeline.steps[1][2]["step_name"] == "library"
        assert pipeline.steps[2][2]["step_name"] == "library"
        assert len(pipeline.images.images) == EXPECTED_PIPELINE_STEPS
        assert isinstance(pipeline.images.images["images_2"].values, np.ndarray)

    def test_special_values_and_callable_resolution(self):
        """Decode tuples and references recursively."""
        params = LibraryLayer.decode_value(
            {"ksize": {"$tuple": [3, 3]}, "borderType": {"$ref": "cv2.BORDER_REFLECT"}}
        )

        assert params == {"ksize": (3, 3), "borderType": cv2.BORDER_REFLECT}
        assert LibraryLayer.resolve_function("skimage", "filters.gaussian").__name__ == "gaussian"

    def test_invalid_symbols_and_results(self):
        """Reject private imports and results that are not images."""
        with pytest.raises(ValueError, match="private symbol"):
            LibraryLayer.resolve_function("skimage", "filters.__dict__")
        with pytest.raises(TypeError, match="not a NumPy image array"):
            LibraryLayer.validate_result(1.0)
