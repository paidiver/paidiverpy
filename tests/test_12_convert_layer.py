"""Tests for the Simple Pipeline class."""

import unittest
import numpy as np
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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[1][0].dtype.itemsize == SIXTEEN_BITS_SIZE
        assert images[2][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[3][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[4][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[1][0].dtype.itemsize == SIXTEEN_BITS_SIZE
        assert images[2][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[3][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        for idx in range(number_images):
            assert images[idx][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[0][0].shape[-1] == NUM_CHANNELS_GREY
        assert images[1][0].shape[-1] == NUM_CHANNELS_RGB
        assert images[2][0].shape[-1] == NUM_CHANNELS_RGBA
        assert images[3][0].shape[-1] == NUM_CHANNELS_GREY
        assert images[4][0].shape[-1] == NUM_CHANNELS_GREY

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        for idx in range(number_images):
            assert images[idx][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[0][0].shape[-1] == NUM_CHANNELS_RGBA
        assert images[1][0].shape[-1] == NUM_CHANNELS_RGB
        assert images[2][0].shape[-1] == NUM_CHANNELS_RGB
        assert images[3][0].shape[-1] == NUM_CHANNELS_RGBA
        assert images[4][0].shape[-1] == NUM_CHANNELS_RGBA
        assert images[5][0].shape[-1] == NUM_CHANNELS_RGB
        assert images[6][0].shape[-1] == NUM_CHANNELS_GREY
        assert images[7][0].shape[-1] == NUM_CHANNELS_RGBA

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[0][0].max() <= EIGHT_BITS_MAX
        assert images[0][0].min() >= 0
        assert images[1][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[1][0].max() <= 1
        assert images[1][0].min() >= 0
        assert images[2][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[2][0].max() <= EIGHT_BITS_MAX
        assert images[2][0].min() >= 0
        assert images[3][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[3][0].max() <= EIGHT_BITS_MAX
        assert images[3][0].min() >= 0

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        assert images[0][0].max() <= EIGHT_BITS_MAX
        assert images[0][0].min() >= 0
        assert images[1][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[1][0].max() <= 1
        assert images[1][0].min() >= 0
        assert images[2][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[2][0].max() <= EIGHT_BITS_MAX
        assert images[2][0].min() >= 0
        assert images[3][0].dtype.itemsize == THIRTY_TWO_BITS_SIZE
        assert images[3][0].max() <= 1
        assert images[3][0].min() >= 0
        assert images[4][0].shape[-1] == NUM_CHANNELS_GREY

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0 = images[0][0].shape
        shape_image_1 = images[1][0].shape
        shape_image_2 = images[2][0].shape
        shape_image_3 = images[3][0].shape
        shape_image_4 = images[4][0].shape
        shape_image_5 = images[5][0].shape
        shape_image_6 = images[6][0].shape
        assert shape_image_1[0] == int(shape_image_0[0] / 2)
        assert shape_image_1[1] == int(shape_image_0[1] / 2)
        assert shape_image_2[0] == int(shape_image_1[0] / 2)
        assert shape_image_2[1] == int(shape_image_1[1] / 2)
        assert shape_image_3[0] == shape_image_2[0]
        assert shape_image_3[1] == shape_image_2[1]
        assert shape_image_4[0] == shape_image_3[0]
        assert shape_image_4[1] == shape_image_3[1]
        assert shape_image_5[0] == 100  # noqa: PLR2004
        assert shape_image_5[1] == 100  # noqa: PLR2004
        assert shape_image_6[0] == 200  # noqa: PLR2004
        assert shape_image_6[1] == 200  # noqa: PLR2004

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0 = images[0][0].shape
        shape_image_1 = images[1][0].shape
        shape_image_2 = images[2][0].shape
        shape_image_3 = images[3][0].shape
        shape_image_4 = images[4][0].shape
        shape_image_5 = images[5][0].shape
        shape_image_6 = images[6][0].shape
        assert shape_image_1[0] == int(shape_image_0[0] / 2)
        assert shape_image_1[1] == int(shape_image_0[1] / 2)
        assert shape_image_2[0] == int(shape_image_1[0] / 2)
        assert shape_image_2[1] == int(shape_image_1[1] / 2)
        assert shape_image_3[0] == shape_image_2[0]
        assert shape_image_3[1] == shape_image_2[1]
        assert shape_image_4[0] == shape_image_3[0]
        assert shape_image_4[1] == shape_image_3[1]
        assert shape_image_5[0] == 100  # noqa: PLR2004
        assert shape_image_5[1] == 100  # noqa: PLR2004
        assert shape_image_6[0] == 200  # noqa: PLR2004
        assert shape_image_6[1] == 200  # noqa: PLR2004

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0 = images[0][0].shape
        shape_image_1 = images[1][0].shape
        shape_image_2 = images[2][0].shape
        shape_image_3 = images[3][0].shape
        shape_image_4 = images[4][0].shape
        shape_image_5 = images[5][0].shape
        shape_image_6 = images[6][0].shape
        shape_image_7 = images[7][0].shape
        assert shape_image_1[0] == int(shape_image_0[0])
        assert shape_image_1[1] == int(shape_image_0[1])
        assert shape_image_2[0] == int(shape_image_1[0] * 0.9)
        assert shape_image_2[1] == int(shape_image_1[1] * 0.9)
        assert shape_image_3[0] == image_100
        assert shape_image_3[1] == image_100
        assert shape_image_4[0] == image_90
        assert shape_image_4[1] == image_90
        assert shape_image_5[0] == image_90
        assert shape_image_5[1] == image_90
        assert shape_image_6[0] == image_90
        assert shape_image_6[1] == image_90
        assert shape_image_7[0] == image_90
        assert shape_image_7[1] == image_90

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
        assert isinstance(images, list)
        assert isinstance(images[0][0], np.ndarray)
        assert images[0][0].dtype.itemsize == EIGHT_BITS_SIZE
        shape_image_0 = images[0][0].shape
        shape_image_1 = images[1][0].shape
        shape_image_2 = images[2][0].shape
        shape_image_3 = images[3][0].shape
        shape_image_4 = images[4][0].shape
        shape_image_5 = images[5][0].shape
        shape_image_6 = images[6][0].shape
        shape_image_7 = images[7][0].shape
        assert shape_image_1[0] == int(shape_image_0[0])
        assert shape_image_1[1] == int(shape_image_0[1])
        assert shape_image_2[0] == int(shape_image_1[0] * 0.9)
        assert shape_image_2[1] == int(shape_image_1[1] * 0.9)
        assert shape_image_3[0] == image_100
        assert shape_image_3[1] == image_100
        assert shape_image_4[0] == image_90
        assert shape_image_4[1] == image_90
        assert shape_image_5[0] == image_90
        assert shape_image_5[1] == image_90
        assert shape_image_6[0] == image_90
        assert shape_image_6[1] == image_90
        assert shape_image_7[0] == image_90
        assert shape_image_7[1] == image_90


if __name__ == "__main__":
    unittest.main()
