"""Base class for custom algorithms."""

import dask.array.core
import numpy as np
from paidiverpy.config.custom_params import CustomParams


class BaseCustomAlgorithm:
    """Base class for custom algorithms.

    Args:
        image_data (np.ndarray | dask.array.core.Array): The image data to process
        params (DynamicConfig): The parameters for the custom algorithm
    """

    def __init__(self, image_data: np.ndarray | dask.array.core.Array, params: CustomParams):
        self.image_data = image_data
        self.params = params

    def process(self) -> np.ndarray | dask.array.core.Array:
        """Process the image data.

        Returns:
            np.ndarray | dask.array.core.Array: The processed image data
        """
        # You must implement this method in your custom algorithm
        msg = "The process method must be implemented"
        raise NotImplementedError(msg)
