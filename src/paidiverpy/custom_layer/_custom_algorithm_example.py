"""This is an example of a custom algorithm that scales the image data using MinMaxScaler from sklearn.preprocessing."""

import ast
import numpy as np
from sklearn import preprocessing
from paidiverpy.custom_layer.base_custom_algorithm import BaseCustomAlgorithm
from paidiverpy.utils.data import NUM_DIMENSIONS
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY


class MyMethod(BaseCustomAlgorithm):
    """This class scales the image data using MinMaxScaler from sklearn.preprocessing."""

    def process(self) -> np.ndarray:
        """This method scales the image data using MinMaxScaler from sklearn.preprocessing.

        Returns:
            tuple[list[np.ndarray], dict]: The dataset with scaled image data and metadata.
        """
        if len(self.image_data.shape) == NUM_DIMENSIONS and self.image_data.shape[-1] == 1:
            self.image_data = np.squeeze(self.image_data, axis=-1)
        feature_range = ast.literal_eval(self.params.feature_range)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        self.image_data = min_max_scaler.fit_transform(self.image_data)
        if len(self.image_data.shape) == NUM_DIMENSIONS_GREY:
            self.image_data = np.expand_dims(self.image_data, axis=-1)
        return self.image_data, self.metadata
