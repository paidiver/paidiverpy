import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from utils import initialise_logging


class ImageLayer:

    def __init__(self, image, image_metadata, step_order, step_name, logger=None):
        self.image = image
        self.image_metadata = image_metadata
        self.step_order = step_order
        self.step_name = step_name
        self.logger = logger or initialise_logging()

    def get_filename(self):
        return self.image_metadata['filename']

    def show(self, subplot=None, title=None):
        if subplot:
            plt.subplot(subplot)
        if title:
            plt.title(title)
        if len(self.image.shape) == 2:
            plt.imshow(self.image, cmap='gray')
        else:
            plt.imshow(self.image)
            
        # cv2.imshow('image', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    def save(self, output_path, filename=None, image_format='png'):
        if isinstance(output_path, str):
            output_path = Path(output_path)
        if not filename:
            filename = f"output_{self.get_filename().split('.')[0]}_{self.step_order}"
        img_path = output_path / f"{filename}.{image_format}"
        cv2.imwrite(str(img_path), self.image)
        self.logger.info("Image saved to %s", img_path)
