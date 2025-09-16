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
from paidiverpy.utils.data import SIXTEEN_BITS_SIZE
from paidiverpy.utils.data import THIRTY_TWO_BITS_SIZE
from tests.base_test_class import BaseTestClass


class TestConvertLayer(BaseTestClass):
    """Tests Simple Pipeline.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_convert_bits_one(self):
        """Test the convert bits step."""
        number_images = 5
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_bits_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_1"].values.dtype.itemsize == SIXTEEN_BITS_SIZE
        assert images["images_2"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_3"].values.dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_4"].values.dtype.itemsize == THIRTY_TWO_BITS_SIZE

    def test_convert_bits_several(self):
        """Test the convert bits step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_bits_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_1"].values.dtype.itemsize == SIXTEEN_BITS_SIZE
        assert images["images_2"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_3"].values.dtype.itemsize == THIRTY_TWO_BITS_SIZE

    def test_convert_to_one(self):
        """Test the convert to step."""
        number_images = 5
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_to_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        for idx in range(number_images):
            assert images[f"images_{idx}"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_1"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_2"].shape[-1] == NUM_CHANNELS_RGBA
        assert images["images_3"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_4"].shape[-1] == NUM_CHANNELS_GREY

    def test_convert_to_several(self):
        """Test the convert to step."""
        number_images = 8
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_to_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        for idx in range(number_images):
            assert images[f"images_{idx}"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].shape[-1] == NUM_CHANNELS_RGBA
        assert images["images_1"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_2"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_3"].shape[-1] == NUM_CHANNELS_RGBA
        assert images["images_4"].shape[-1] == NUM_CHANNELS_RGBA
        assert images["images_5"].shape[-1] == NUM_CHANNELS_RGB
        assert images["images_6"].shape[-1] == NUM_CHANNELS_GREY
        assert images["images_7"].shape[-1] == NUM_CHANNELS_RGBA

    def test_convert_normalise_one(self):
        """Test the convert normalise step."""
        number_images = 4
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_normalise_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"].values.dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"].values.max() <= EIGHT_BITS_MAX
        assert images["images_0"].values.min() >= 0
        assert images["images_1"].values.dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_1"].values.max() <= 1
        assert images["images_1"].values.min() >= 0
        assert images["images_2"].values.dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_2"].values.max() <= EIGHT_BITS_MAX
        assert images["images_2"].values.min() >= 0
        assert images["images_3"].values.dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_3"].values.max() <= EIGHT_BITS_MAX
        assert images["images_3"].values.min() >= 0

    def test_convert_normalise_several(self):
        """Test the convert normalise step."""
        number_images = 5
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_normalise_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images["images_0"][0].max() <= EIGHT_BITS_MAX
        assert images["images_0"][0].min() >= 0
        assert images["images_1"][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_1"][0].max() <= 1
        assert images["images_1"][0].min() >= 0
        assert images["images_2"][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_2"][0].max() <= EIGHT_BITS_MAX
        assert images["images_2"][0].min() >= 0
        assert images["images_3"][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images["images_3"][0].max() <= 1
        assert images["images_3"][0].min() >= 0
        assert images["images_4"][0].shape[-1] == NUM_CHANNELS_GREY

    def test_convert_resize_one(self):
        """Test the convert resize step."""
        number_images = 8
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_resize_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0_h = images["original_height_0"][0]
        shape_image_1_h = images["original_height_1"][0]
        shape_image_2_h = images["original_height_2"][0]
        shape_image_3_h = images["original_height_3"][0]
        shape_image_4_h = images["original_height_4"][0]
        shape_image_5_h = images["original_height_5"][0]
        shape_image_6_h = images["original_height_6"][0]
        shape_image_0_w = images["original_width_0"][0]
        shape_image_1_w = images["original_width_1"][0]
        shape_image_2_w = images["original_width_2"][0]
        shape_image_3_w = images["original_width_3"][0]
        shape_image_4_w = images["original_width_4"][0]
        shape_image_5_w = images["original_width_5"][0]
        shape_image_6_w = images["original_width_6"][0]
        assert shape_image_1_h == int(shape_image_0_h / 2)
        assert shape_image_1_w == int(shape_image_0_w / 2)
        assert shape_image_2_h == int(shape_image_1_h / 2)
        assert shape_image_2_w == int(shape_image_1_w / 2)
        assert shape_image_3_h == shape_image_2_h
        assert shape_image_3_w == shape_image_2_w
        assert shape_image_4_h == shape_image_3_h
        assert shape_image_4_w == shape_image_3_w
        assert shape_image_5_h == 100  # noqa: PLR2004
        assert shape_image_5_w == 100  # noqa: PLR2004
        assert shape_image_6_h == 200  # noqa: PLR2004
        assert shape_image_6_w == 200  # noqa: PLR2004

    def test_convert_resize_several(self):
        """Test the convert resize step for several channel images."""
        number_images = 8
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_resize_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0_h = images["original_height_0"][0]
        shape_image_1_h = images["original_height_1"][0]
        shape_image_2_h = images["original_height_2"][0]
        shape_image_3_h = images["original_height_3"][0]
        shape_image_4_h = images["original_height_4"][0]
        shape_image_5_h = images["original_height_5"][0]
        shape_image_6_h = images["original_height_6"][0]
        shape_image_0_w = images["original_width_0"][0]
        shape_image_1_w = images["original_width_1"][0]
        shape_image_2_w = images["original_width_2"][0]
        shape_image_3_w = images["original_width_3"][0]
        shape_image_4_w = images["original_width_4"][0]
        shape_image_5_w = images["original_width_5"][0]
        shape_image_6_w = images["original_width_6"][0]
        assert shape_image_1_h == int(shape_image_0_h / 2)
        assert shape_image_1_w == int(shape_image_0_w / 2)
        assert shape_image_2_h == int(shape_image_1_h / 2)
        assert shape_image_2_w == int(shape_image_1_w / 2)
        assert shape_image_3_h == shape_image_2_h
        assert shape_image_3_w == shape_image_2_w
        assert shape_image_4_h == shape_image_3_h
        assert shape_image_4_w == shape_image_3_w
        assert shape_image_5_h == 100  # noqa: PLR2004
        assert shape_image_5_w == 100  # noqa: PLR2004
        assert shape_image_6_h == 200  # noqa: PLR2004
        assert shape_image_6_w == 200  # noqa: PLR2004

    def test_convert_crop_one(self):
        """Test the convert crop step."""
        image_100 = 100
        image_90 = 90
        number_images = 8
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_crop_one.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0_h = images["original_height_0"][0]
        shape_image_1_h = images["original_height_1"][0]
        shape_image_2_h = images["original_height_2"][0]
        shape_image_3_h = images["original_height_3"][0]
        shape_image_4_h = images["original_height_4"][0]
        shape_image_5_h = images["original_height_5"][0]
        shape_image_6_h = images["original_height_6"][0]
        shape_image_7_h = images["original_height_7"][0]
        shape_image_0_w = images["original_width_0"][0]
        shape_image_1_w = images["original_width_1"][0]
        shape_image_2_w = images["original_width_2"][0]
        shape_image_3_w = images["original_width_3"][0]
        shape_image_4_w = images["original_width_4"][0]
        shape_image_5_w = images["original_width_5"][0]
        shape_image_6_w = images["original_width_6"][0]
        shape_image_7_w = images["original_width_7"][0]
        assert shape_image_1_h == int(shape_image_0_h)
        assert shape_image_1_w == int(shape_image_0_w)
        assert shape_image_2_h == int(shape_image_1_h * 0.9)
        assert shape_image_2_w == int(shape_image_1_w * 0.9)
        assert shape_image_3_h == image_100
        assert shape_image_3_w == image_100
        assert shape_image_4_h == image_90
        assert shape_image_4_w == image_90
        assert shape_image_5_h == image_90
        assert shape_image_5_w == image_90
        assert shape_image_6_h == image_90
        assert shape_image_6_w == image_90
        assert shape_image_7_h == image_90
        assert shape_image_7_w == image_90

    def test_convert_crop_several(self):
        """Test the convert crop step."""
        image_100 = 100
        image_90 = 90
        number_images = 8
        pipeline = Pipeline(config_file_path="tests/config_files/convert_layer/config_convert_crop_several.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images["images_0"], xr.DataArray)
        assert isinstance(images["images_0"].values, np.ndarray)
        assert images["images_0"][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0_h = images["original_height_0"][0]
        shape_image_1_h = images["original_height_1"][0]
        shape_image_2_h = images["original_height_2"][0]
        shape_image_3_h = images["original_height_3"][0]
        shape_image_4_h = images["original_height_4"][0]
        shape_image_5_h = images["original_height_5"][0]
        shape_image_6_h = images["original_height_6"][0]
        shape_image_7_h = images["original_height_7"][0]
        shape_image_0_w = images["original_width_0"][0]
        shape_image_1_w = images["original_width_1"][0]
        shape_image_2_w = images["original_width_2"][0]
        shape_image_3_w = images["original_width_3"][0]
        shape_image_4_w = images["original_width_4"][0]
        shape_image_5_w = images["original_width_5"][0]
        shape_image_6_w = images["original_width_6"][0]
        shape_image_7_w = images["original_width_7"][0]
        assert shape_image_1_h == int(shape_image_0_h)
        assert shape_image_1_w == int(shape_image_0_w)
        assert shape_image_2_h == int(shape_image_1_h * 0.9)
        assert shape_image_2_w == int(shape_image_1_w * 0.9)
        assert shape_image_3_h == image_100
        assert shape_image_3_w == image_100
        assert shape_image_4_h == image_90
        assert shape_image_4_w == image_90
        assert shape_image_5_h == image_90
        assert shape_image_5_w == image_90
        assert shape_image_6_h == image_90
        assert shape_image_6_w == image_90
        assert shape_image_7_h == image_90
        assert shape_image_7_w == image_90


if __name__ == "__main__":
    unittest.main()
