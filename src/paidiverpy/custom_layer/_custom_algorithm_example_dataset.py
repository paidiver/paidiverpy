"""This is an example of a custom algorithm that scales the image data using MinMaxScaler from sklearn.preprocessing."""

import ast
import numpy as np
from sklearn import preprocessing
from paidiverpy.custom_layer.base_custom_algorithm import BaseCustomAlgorithm
from paidiverpy.utils.data import NUM_DIMENSIONS
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY


class MyMethod(BaseCustomAlgorithm):
    """This class scales all the images in the dataset using MinMaxScaler from sklearn.preprocessing."""

    def process(self) -> tuple[list[np.ndarray], dict]:
        """This method scales all the images in the dataset using MinMaxScaler from sklearn.preprocessing.

        Returns:
            tuple[list[np.ndarray], dict]: The dataset with scaled image data and metadata.
        """
        feature_range = ast.literal_eval(self.params.feature_range)
        for idx, image in enumerate(self.image_data):
            img = np.squeeze(image, axis=-1) if image.ndim == NUM_DIMENSIONS and image.shape[-1] == 1 else image
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
            img = min_max_scaler.fit_transform(img)
            if len(img.shape) == NUM_DIMENSIONS_GREY:
                img = np.expand_dims(img, axis=-1)
            self.image_data[idx] = img
        return self.image_data, self.metadata
