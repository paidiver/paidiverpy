"""This is an example of a custom algorithm that scales the image data using MinMaxScaler from sklearn.preprocessing."""

import ast
import xarray as xr
from sklearn import preprocessing
from paidiverpy.custom_layer import CustomLayer
from paidiverpy.models.custom_params import CustomParams


class MyDatasetCustomClass(CustomLayer):
    """This class scales all the images in the dataset using MinMaxScaler from sklearn.preprocessing."""

    def min_max_data_dataset(self, images: xr.Dataset, params: CustomParams | None = None) -> xr.Dataset:
        """This method scales all the images in the dataset using MinMaxScaler from sklearn.preprocessing.

        Args:
            images (xr.Dataset): The dataset containing the images to be scaled.
            params (CustomParams, optional): The parameters for the custom algorithm.

        Returns:
            xr.Dataset: The updated dataset with scaled images.
        """
        feature_range = ast.literal_eval(params.feature_range)
        images_da = images["images"]
        images_data = images_da.squeeze("band").data if images_da.sizes["band"] == 1 else images_da.data

        n_files = images_da.sizes["filename"]
        y = images_da.sizes["y"]
        x = images_da.sizes["x"]
        bands = images_da.sizes["band"] if "band" in images_da.dims else 1

        flat_images = images_data.reshape(n_files, y * x * bands)

        scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        flat_images_scaled = scaler.fit_transform(flat_images)

        images_scaled = flat_images_scaled.reshape(n_files, y, x, bands)

        images["images"].data = images_scaled

        return images
