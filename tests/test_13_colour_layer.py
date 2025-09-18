"""Tests for the Simple Pipeline class."""

import unittest
import numpy as np
import xarray as xr
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from paidiverpy.utils.data import EIGHT_BITS_MAX
from paidiverpy.utils.data import EIGHT_BITS_SIZE
from paidiverpy.utils.data import NUM_CHANNELS_GREY
from paidiverpy.utils.data import NUM_CHANNELS_RGB
from paidiverpy.utils.data import NUM_CHANNELS_RGBA
from tests.base_test_class import BaseTestClass


class TestColourLayer(BaseTestClass):
    """Tests Simple Pipeline.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_colour_grayscale_several(self):
        """Test the colour grayscale step."""
        number_images = 11
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_grayscale_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].shape[-1] == NUM_CHANNELS_RGBA
        assert images["images_1"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_2"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_3"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_4"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_5"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_6"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_7"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_8"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_9"].shape[-1] == NUM_CHANNELS_GREY
        assert (images["images_7"].values[0][0][0] + images["images_9"].values[0][0][0]) == EIGHT_BITS_MAX
        assert images["images_10"].shape[-1] == NUM_CHANNELS_GREY

    def test_colour_grayscale_one(self):
        """Test the colour greyscale step."""
        number_images = 10
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_grayscale_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"][0].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_1"][0].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_2"][0].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_3"][0].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_4"][0].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_5"][0].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_6"][0].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_7"][0].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_8"][0].shape[-1] == NUM_CHANNELS_GREY
        assert (images["images_6"].values[0][0][0] + images["images_8"].values[0][0][0]) == EIGHT_BITS_MAX
        assert images["images_9"][0].shape[-1] == NUM_CHANNELS_GREY

    def test_colour_gaussian_blur_one(self):
        """Test the colour gaussian blur step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_gaussian_blur_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() != images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() != images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()

    def test_colour_gaussian_blur_several(self):
        """Test the colour gaussian blur step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_gaussian_blur_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() != images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() != images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()

    def test_colour_sharpen_one(self):
        """Test the colour sharpen step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_sharpen_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() > images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() > images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()

    def test_colour_sharpen_several(self):
        """Test the colour sharpen step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_sharpen_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() > images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() > images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()

    def test_colour_contrast_one(self):
        """Test the colour contrast step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_contrast_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() < images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() < images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()

    def test_colour_contrast_several(self):
        """Test the colour contrast step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_contrast_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() != images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() != images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()

    def test_colour_illumination_one(self):
        """Test the colour illumination step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_illumination_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"][0].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"][0].mean() != images["images_1"][0].mean()
        assert images["images_1"][0].mean() != images["images_2"][0].mean()
        assert images["images_2"][0].mean() == images["images_3"][0].mean()

    def test_colour_illumination_several(self):
        """Test the colour illumination step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_illumination_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"][0].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"][0].mean() != images["images_1"][0].mean()
        assert images["images_1"][0].mean() != images["images_2"][0].mean()
        assert images["images_2"][0].mean() == images["images_3"][0].mean()

    def test_colour_deblur_one(self):
        """Test the colour deblur step."""
        number_images = 5
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_deblur_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() != images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() != images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() != images["images_3"].values[0].mean()
        assert images["images_3"].values[0].mean() == images["images_4"].values[0].mean()

    def test_colour_deblur_several(self):
        """Test the colour deblur step."""
        number_images = 5
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_deblur_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() != images["images_1"].values[0].mean()
        # assert images["images_1"].values[0].mean() != images["images_2"].values[0].mean()
        # assert images["images_2"].values[0].mean() != images["images_3"].values[0].mean()
        assert images["images_3"].values[0].mean() == images["images_4"].values[0].mean()

    def test_colour_alteration_one(self):
        """Test the colour colour alteration step."""
        number_images = 7
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_colour_alteration_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE

    def test_colour_alteration_several(self):
        """Test the colour colour alteration step."""
        number_images = 5
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_colour_alteration_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values[0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values[0].mean() != images["images_1"].values[0].mean()
        assert images["images_1"].values[0].mean() != images["images_2"].values[0].mean()
        assert images["images_2"].values[0].mean() == images["images_3"].values[0].mean()
        assert images["images_3"].values[0].mean() == images["images_4"].values[0].mean()

    def test_edge_one(self):
        """Test the colour edge step."""
        number_images = 7
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_edge_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"][0].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"][0].values.mean() != images["images_1"][0].values.mean()
        assert images["images_1"][0].values.mean() == images["images_2"][0].values.mean()
        assert images["images_2"][0].values.mean() != images["images_3"][0].values.mean()
        assert images["images_3"][0].values.mean() != images["images_4"][0].values.mean()
        assert images["images_4"][0].values.mean() != images["images_5"][0].values.mean()
        # assert images["images_5"][1].values.mean() == images["images_6"][1].values.mean()

    def test_edge_several(self):
        """Test the colour edge step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/colour_layer/config_colour_edge_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images, xr.Dataset)
        assert isinstance(images["images_0"][0].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"][0].values.mean() != images["images_1"][0].values.mean()
        assert images["images_1"][0].values.mean() != images["images_2"][0].values.mean()


if __name__ == "__main__":
    unittest.main()
