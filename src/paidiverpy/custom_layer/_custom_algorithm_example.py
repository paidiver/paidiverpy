"""This is an example of a custom algorithm that scales the image data using MinMaxScaler from sklearn.preprocessing."""

import ast
from typing import Any
import numpy as np
from sklearn import preprocessing
from paidiverpy.custom_layer import CustomLayer
from paidiverpy.models.custom_params import CustomParams
from paidiverpy.utils.data import NUM_DIMENSIONS
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY


class MyCustomClass(CustomLayer):
    """This class scales the image data using MinMaxScaler from sklearn.preprocessing."""

    @staticmethod
    def min_max_data(image_data: np.ndarray[Any, Any], params: CustomParams | None = None, **_kwargs: dict[str, Any]) -> np.ndarray[Any, Any]:
        """Convert the image to the specified number of bits.

        Args:
            image_data (xr.DataArray): The image data.
            params (CustomParams, optional): The custom parameters.
            **_kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The updated image.
        """
        if len(image_data.shape) == NUM_DIMENSIONS and image_data.shape[-1] == 1:
            image_data = np.squeeze(image_data, axis=-1)
        feature_range = ast.literal_eval(params.feature_range)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        image_data = min_max_scaler.fit_transform(image_data)
        if len(image_data.shape) == NUM_DIMENSIONS_GREY:
            image_data = np.expand_dims(image_data, axis=-1)
        return image_data
