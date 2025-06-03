"""Base class for custom algorithms."""

import dask.array.core
import numpy as np
import pandas as pd
from paidiverpy.models.custom_params import CustomParams


class BaseCustomAlgorithm:
    """Base class for custom algorithms.

    Args:
        image_data (np.ndarray | dask.array.core.Array | list[np.ndarray]): The image data to process (individual image or dataset)
        metadata (dict): The metadata for the image data (individual image or dataset)
        params (CustomParams): The parameters for the custom algorithm
        metadata_object (pd.DataFrame, optional): The metadata object for the image data.
    It is not required for datasets, but it is required for individual images.
    """

    def __init__(
        self,
        image_data: np.ndarray | dask.array.core.Array,
        metadata: dict,
        params: CustomParams,
        metadata_core: pd.DataFrame | None = None,
    ):
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
