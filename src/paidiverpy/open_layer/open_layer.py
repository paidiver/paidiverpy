""" Open raw image file
"""
import gc
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import cv2
from tqdm import tqdm
from paidiverpy import Paidiverpy
from paidiverpy.convert_layer import ConvertLayer
from paidiverpy.image_layer import ImageLayer
from paidiverpy.resample_layer import ResampleLayer

class OpenLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None,
                 logger=None,
                 images=None,
                 paidiverpy=None,
                 step_name='raw',
                 parameters=None,
                 raise_error=False,
                 verbose=True):

        super().__init__(config_file_path=config_file_path,
                         input_path=input_path,
                         output_path=output_path,
                         catalog_path=catalog_path,
                         catalog_type=catalog_type,
                         catalog=catalog,
                         config=config,
                         logger=logger,
                         images=images,
                         paidiverpy=paidiverpy,
                         raise_error=raise_error,
                         verbose=verbose)

        self.step_name = step_name
        if parameters:
            self.config.add_config('general', parameters)

        self.extract_exif()
        self.step_metadata = self._calculate_steps_metadata(self.config.general)

    def run(self):
        if self.step_name == 'raw':
            self.import_image()

    def import_image(self):
        if self.step_metadata.get('sampling_mode') and self.step_metadata.get('sampling_limits'):
            self.config.general.mode = self.step_metadata.get('sampling_mode')
            self.config.general.limits = self.step_metadata.get('sampling_limits')
            self.set_catalog(ResampleLayer(config=self.config, catalog=self.catalog).run())

        img_path_list = [
            self.config.general.input_path / filename for filename in self.get_catalog()['filename']]

        image_list = []

        for index, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list), desc="Processing Images"):
            img = self.open_image(
                img_path=img_path,
                image_metadata=self.get_catalog(flag='all').iloc[index].to_dict()
            )
            image_list.append(img)
            del img
            gc.collect()
        self.images.add_step(step=self.step_name, images=image_list, step_metadata=self.step_metadata)
        del image_list
        gc.collect()

    def open_image(self,
                   img_path,
                   image_metadata = None):

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if self.step_metadata.get('convert_bayer_pattern'):
            try:
                bayer_pattern = ConvertLayer.get_bayer_pattern(
                    img=img,
                    bayer_pattern=self.step_metadata.get('convert_bayer_pattern'),
                    logger=self.logger,
                    raise_error=self.raise_error)
                img = cv2.cvtColor(img, bayer_pattern)
            except Exception as e:
                if self.raise_error:
                    self.logger.error("Failed to convert the image to the Bayer pattern: %s", str(e))
                    raise ValueError("Failed to convert the image to the Bayer pattern") from e
                self.logger.warning("Failed to convert the image to the Bayer pattern: %s", str(e))
                self.logger.warning("The image will be processed without the Bayer pattern conversion")                
                return img
        
        if self.step_metadata.get('convert_bits'):
            img = ConvertLayer.convert_bits(img,
                                            self.step_metadata.get('convert_bits'),
                                            self.step_metadata.get('convert_autoscale'),
                                            logger=self.logger,
                                            raise_error=self.raise_error)
        if self.step_metadata.get('convert_to'):
            img = ConvertLayer.channel_convert(
                img,
                self.step_metadata.get('convert_to'),
                self.step_metadata.get('convert_channel_selector'),
                logger=self.logger,
                raise_error=self.raise_error)
        # img = OpenLayer.normalize_image(img)
        if self.step_metadata.get('convert_normalize'):
            img = ConvertLayer.normalize_image(img,
                                               self.step_metadata['convert_normalize'],
                                               logger=self.logger,
                                               raise_error=self.raise_error)
        img = ImageLayer(image=img,
                         image_metadata=image_metadata,
                         step_order=self.images.get_last_step_order(),
                         step_name=self.step_name)
        return img

    def extract_exif(self):
        img_path_list = [self.config.general.input_path / filename for filename in self.get_catalog()['filename']]
        exif_list = []
        for img_path in img_path_list:
            exif_list.append(OpenLayer.extract_exif_single(img_path))
        self.set_catalog(self.get_catalog(flag='all').merge(pd.DataFrame(exif_list), on='filename', how='left'))

    @staticmethod
    def extract_exif_single(img_path):
        img_pil = Image.open(img_path)
        exif_data = img_pil.getexif()
        exif = {}
        if exif_data is not None:
            exif['filename'] = img_path.name
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif[tag_name] = value
        return exif
