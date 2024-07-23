""" Open raw image file
"""
import numpy as np
import cv2
from paidiverpy import Paidiverpy

class ConvertLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None):
        super().__init__(config_file_path, input_path, output_path, catalog_path, catalog_type, catalog, config)
            
    def convert_bits(self, img, bits = 8):
        if self.config.general.autoscale:
            try:
                result = np.float32(img) - np.min(img)
                result[result < 0.0] = 0.0
                if np.max(img) != 0:
                    result = result / np.max(img)
                if bits == 8:
                    img_bit = np.uint8(255 * result)
                elif bits == 16:
                    img_bit = np.uint16(65535 * result)
                elif bits == 32:
                    img_bit = np.float32(result)
            except Exception as e:
                print(e)
                print(f"Failed to convert the image to 8-bit: {str(e)}")
                img_bit = None
        else:
            if bits == 8:
                img_bit = np.uint8(255)
            elif bits == 16:
                img_bit = np.uint16(65535)
            elif bits == 32:
                img_bit = np.float32(result)
        return img_bit

    def generate_output(self, img, image_format = 'png'):
        img_path = self.output_path / f"{img['filename']}.{image_format}"
        cv2.imwrite(str(img_path), img['image'])
        print(f"Image saved to {img_path}")

