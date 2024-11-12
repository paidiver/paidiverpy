from typing import Union
import numpy as np
import dask.array.core

class BaseAlgorithm:
    def __init__(self, image_data: Union[np.ndarray, dask.array.core.Array], params: dict):
        self.image_data = image_data
        self.params = params

    def process(self) -> Union[np.ndarray, dask.array.core.Array]:
        # You must implement this method in your custom algorithm
        raise NotImplementedError("The process method must be implemented")
