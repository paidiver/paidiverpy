""" ImageLayer class to represent an image layer.
"""
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np

from utils import initialise_logging


class ImageLayer:
    """ ImageLayer class to represent an image layer.
    
    Args:
        image (np.ndarray): The image data.
        image_metadata (dict): The image metadata.
        step_order (int): The order of the step in the pipeline.
        step_name (str): The name of the step in the pipeline.
        logger (logging.Logger): The logger object.   
    """

    def __init__(self,
                 image: np.ndarray,
                 image_metadata: dict,
                 step_order: int,
                 step_name: str,
                 logger: logging.Logger = None):
        self.image = image
        self.image_metadata = image_metadata
        self.step_order = step_order
        self.step_name = step_name
        self.logger = logger or initialise_logging()

    def get_filename(self) -> str:
        """ Get the filename of the image.

        Returns:
            str: The filename of the image.
        """
        return self.image_metadata["filename"]

    def show(self,
             subplot: int = None,
             title: str = None):
        """ Show the image.

        Args:
            subplot (int): The subplot number.
            title (str): The title of the image.
        """
        if subplot:
            plt.subplot(subplot)
        if title:
            plt.title(title)
        if len(self.image.shape) == 2:
            plt.imshow(self.image, cmap="gray")
        else:
            plt.imshow(self.image)

        # cv2.imshow('image', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def save(self, output_path: str, filename: str=None, image_format: str="png"):
        """ Save the image.

        Args:
            output_path (str): The output path.
            filename (str, optional): The filename. Defaults to None.
            image_format (str, optional): The image format. Defaults to "png".
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not filename:
            filename = f"output_{self.get_filename().split('.')[0]}_{self.step_order}"
        img_path = output_path / f"{filename}.{image_format}"
        cv2.imwrite(str(img_path), self.image)
        self.logger.info("Image saved to %s", img_path)
