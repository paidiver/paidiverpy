"""Base class for custom algorithms."""

import dask.array.core
import numpy as np
import pandas as pd
from paidiverpy.config.custom_params import CustomParams


class BaseCustomAlgorithm:
    """Base class for custom algorithms.

    Args:
        image_data (np.ndarray | dask.array.core.Array): The image data to process
        metadata (dict): The metadata for the image data
        params (CustomParams): The parameters for the custom algorithm
        metadata_object (pd.DataFrame, optional): The metadata object for the image data.
    """

    def __init__(self, image_data: np.ndarray | dask.array.core.Array, metadata: dict, params: CustomParams, metadata_core: pd.DataFrame):
        self.image_data = image_data
        self.params = params
        self.metadata = metadata
        self.metadata_core = metadata_core

    def process(self) -> np.ndarray | dask.array.core.Array:
        """Process the image data.

        Returns:
            np.ndarray | dask.array.core.Array: The processed image data
        """
        # You must implement this method in your custom algorithm
        msg = "The process method must be implemented"
        raise NotImplementedError(msg)
