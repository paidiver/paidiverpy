""" Open raw image file
"""
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import cv2
from paidiverpy import Paidiverpy
from paidiverpy.catalog_parser import CatalogParser

class OpenLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None):
        super().__init__(config_file_path, input_path, output_path, catalog_path, catalog_type, catalog, config)
        self.images = None
        self.extract_exif()

    def import_image(self, filenames = None):
        file_names = filenames if filenames else self.catalog['filename']
        img_path_list = [self.input_path / filename for filename in file_names]
        img_flag = cv2.IMREAD_GRAYSCALE if self.config.general.channels == 1 else cv2.IMREAD_UNCHANGED
        bayer_pattern = None
        # if self.config.cv_attribute.is_raw and self.config.cv_attribute.channels != 1:
        #     bayer_pattern = self.get_bayer_pattern()
        self.images = [self.open_image(img_path, img_flag, bayer_pattern) for img_path in img_path_list]
        # self.images = [self.convert_to_8bit(img) for img in self.images]

    def open_image(self, img_path, img_flag, bayer_pattern):
        try:
            img_c = cv2.imread(img_path, img_flag)
            if bayer_pattern is not None:
                img_c = cv2.cvtColor(img_c, bayer_pattern)
        except Exception as e:
            print(f"Failed to load or convert the image: {str(e)}")
            img_c = None
        return img_c

    def extract_exif(self):
        img_path_list = [self.input_path / filename for filename in self.catalog['filename']]
        exif_list = []
        for img_path in img_path_list:
            exif_list.append(self.extract_exif_single(img_path))
        self.catalog = self.catalog.merge(pd.DataFrame(exif_list), on='filename', how='left')

    def extract_exif_single(self, img_path):
        img_pil = Image.open(img_path)
        exif_data = img_pil.getexif()
        exif = {}
        if exif_data is not None:
            exif['filename'] = img_path.name
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif[tag_name] = value
        return exif

    def get_bayer_pattern(self):
        if self.config.general.channels == 1:
            return None
        elif self.config.general.bayer_pattern == 'RG':
            bayer_pattern = cv2.COLOR_BAYER_RG2RGB
        elif self.config.general.bayer_pattern == 'BG':
            bayer_pattern = cv2.COLOR_BAYER_BG2RGB
        elif self.config.general.bayer_pattern == 'GR':
            bayer_pattern = cv2.COLOR_BAYER_GR2RGB
        else:
            bayer_pattern = cv2.COLOR_BAYER_GB2RGB

        return bayer_pattern
