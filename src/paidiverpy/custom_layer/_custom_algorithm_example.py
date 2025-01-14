"""This is an example of a custom algorithm that scales the image data using MinMaxScaler from sklearn.preprocessing."""

import ast
import numpy as np
from sklearn import preprocessing
from paidiverpy.custom_layer.base_custom_algorithm import BaseCustomAlgorithm


class MyMethod(BaseCustomAlgorithm):
    """This class scales the image data using MinMaxScaler from sklearn.preprocessing."""

    def process(self) -> np.ndarray:
        """This method scales the image data using MinMaxScaler from sklearn.preprocessing.

        Returns:
            np.ndarray: The scaled image data.
        """
        feature_range = ast.literal_eval(self.params.feature_range)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        self.image_data = min_max_scaler.fit_transform(self.image_data)
        return self.image_data
