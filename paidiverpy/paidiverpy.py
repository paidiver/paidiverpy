from skimage import io, exposure
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel, unsharp_mask
from skimage.exposure import rescale_intensity 

class Paidiverpy:
    def __init__(self, data_path: str, file_pattern: str = None, image_t):
        self.image_path = image_path
        self.image = io.imread(image_path)
        self.processed_images = [self.image]

    def process_image(self, steps):
        for step in steps:
            print(globals())
            function = globals()[step['function']]
            if step.get('params'):
                self.image = function(self.image, **step.get('params'))
            else:
                self.image = function(self.image)
            self.processed_images.append(self.image)
