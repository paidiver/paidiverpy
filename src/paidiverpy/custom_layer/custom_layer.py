
"""Color layer module.

This module contains the ColorLayer class for processing the images in the
color layer.
"""

import logging
import importlib
import dask
import dask.array as da
import numpy as np
from dask import compute
from dask.diagnostics import ProgressBar
from scipy import ndimage
from skimage import color
from skimage import measure
from skimage import morphology
from skimage import restoration
from skimage.exposure import adjust_gamma
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.filters import scharr
from skimage.filters import unsharp_mask
from skimage.restoration import rolling_ball
from skimage.restoration import wiener
from skimage.segmentation import checkerboard_level_set
from skimage.segmentation import morphological_chan_vese
from skimage.transform import resize
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils import DynamicConfig
from paidiverpy.utils import raise_value_error

NUM_CHANNELS_RGB = 3
NUM_CHANNELS_RGBA = 4
NUM_IMAGE_DIMS = 2
DEFAULT_BITS = 8

class CustomLayer(Paidiverpy):
    """ColorLayer class.

    Process the images in the color layer.

    Args:
        config_file_path (str): The path to the configuration file.
        input_path (str): The path to the input files.
        output_path (str): The path to the output files.
        metadata_path (str): The path to the metadata file.
        metadata_type (str): The type of the metadata file.
        metadata (MetadataParser): The metadata object.
        config (Configuration): The configuration object.
        logger (logging.Logger): The logger object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        step_name (str): The name of the step.
        parameters (dict): The parameters for the step.
        config_index (int): The index of the configuration.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
        track_changes (bool): Whether to track changes. Defaults to True.
        n_jobs (int): The number of jobs to run in parallel.
    """

    def __init__(
        self,
        config_file_path: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        metadata_path: str | None = None,
        metadata_type: str | None = None,
        metadata: MetadataParser = None,
        config: Configuration = None,
        logger: logging.Logger | None = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        parameters: dict | None = None,
        config_index: int | None = None,
        raise_error: bool = False,
        verbose: int = 2,
        track_changes: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata_type=metadata_type,
            metadata=metadata,
            config=config,
            logger=logger,
            images=images,
            paidiverpy=paidiverpy,
            raise_error=raise_error,
            verbose=verbose,
            track_changes=track_changes,
            n_jobs=n_jobs,
        )

        self.step_name = step_name
        if parameters:
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])

    def run(self, add_new_step: bool = True) -> ImagesLayer | None:
        """Custom Layer run method.

        Run the custom layer steps on the images based on the configuration
        file or parameters.

        Args:
            add_new_step (bool, optional): Whether to add a new step to the images object.
        Defaults to True.

        Raises:
            ValueError: The mode is not defined in the configuration file.

        Returns:
            Union[ImagesLayer, None]: The images object with the new step added.
        """
        algorithm_name = self.step_metadata.get('name')
        module_name = self.step_metadata.get('module')
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method = self.load_custom_algorithm(module_name, algorithm_name)
        images = self.images.get_step(step=len(self.images.images) - 1, by_order=True)
        if self.n_jobs == 1:
            image_list = self.process_sequentially(images, method, params)
        else:
            image_list = self.process_parallel(images, method, params)
        if not test:
            self.step_name = f"custom_{self.config_index}" if not self.step_name else self.step_name
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=image_list,
                    step_metadata=self.step_metadata,
                    metadata=self.get_metadata(),
                    track_changes=self.track_changes,
                )
                return None
            self.images.images[-1] = image_list
            return self.images
        return None

    def process_sequentially(self, images: list[np.ndarray], method: callable, params: dict) -> list[np.ndarray]:
        """Process the images sequentially.

        Method to process the images sequentially.

        Args:
            images (List[np.ndarray]): The list of images to process.
            method (callable): The method to apply to the images.
            params (dict): The parameters for the method.

        Returns:
            List[np.ndarray]: The list of processed images.
        """
        return [method(img, params=params).process() for img in images]

    def process_parallel(
        self, images: list[da.core.Array], method: callable, params: DynamicConfig,
    ) -> list[np.ndarray]:
        """Process the images in parallel.

        Method to process the images in parallel.

        Args:
            images (List[da.core.Array]): The list of images to process.
            method (callable): The method to apply to the images.
            params (DynamicConfig): The parameters for the method.

        Returns:
            List[da.core.Array]: The list of processed images.
        """
        delayed_images = [dask.delayed(lambda img: method(img, params=params).process())(img) for img in images]
        with dask.config.set(scheduler="threads", num_workers=self.n_jobs):
            self.logger.info("Processing images using %s cores", self.n_jobs)
            with ProgressBar():
                delayed_images = compute(*delayed_images)
        return [da.from_array(img) for img in delayed_images]

    def load_custom_algorithm(self, module_name: str, class_name: str) -> callable:
        """ Load a custom algorithm class.

        Args:
            module_name (str): The module name.
            class_name (str): The class name.

        Returns:
            class: The custom algorithm class.
        """
        module = importlib.import_module(module_name)
        algorithm_class = getattr(module, class_name)
        return algorithm_class
