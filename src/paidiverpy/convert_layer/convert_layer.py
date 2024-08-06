""" Open raw image file
"""
import numpy as np
import cv2
from tqdm import tqdm

from paidiverpy import Paidiverpy
from paidiverpy.image_layer import ImageLayer

class ConvertLayer(Paidiverpy):
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
                 step_name=None,
                 parameters=None,
                 config_index=None,
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
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])


    def run(self):
        test = self.step_metadata.get('test')
        images = self.images.get_step(step=len(self.images.images)-1, by_order=True)
        image_list = []
        for index, img in tqdm(enumerate(images), total=len(images), desc="Processing Images"):
            img_data = img.image
            if self.step_metadata.get('bits'):
                img_data = ConvertLayer.convert_bits(img_data, output_bits=self.step_metadata.get('bits'), autoscale=self.step_metadata.get('autoscale'))
            if self.step_metadata.get('to'):
                img_data = ConvertLayer.channel_convert(img_data, to=self.step_metadata.get('to'), channel_selector=self.step_metadata.get('channel_selector'))
            if self.step_metadata.get('bayer_pattern'):
                try:
                    bayer_pattern = ConvertLayer.get_bayer_pattern(
                        img=img_data,
                        bayer_pattern=self.step_metadata.get('bayer_pattern'),
                        logger=self.logger,
                        raise_error=self.raise_error)
                    img_data = cv2.cvtColor(img_data, bayer_pattern)
                except Exception as e:
                    if self.raise_error:
                        self.logger.error("Failed to convert the image to the Bayer pattern: %s", str(e))
                        raise ValueError(f"Failed to convert the image to the Bayer pattern: {str(e)}") from e
                    self.logger.warning("Failed to convert the image to the Bayer pattern: %s", str(e))
                    self.logger.warning("The image will be processed without the Bayer pattern conversion.")
                    return img_data
            if self.step_metadata.get('normalize'):
                img_data = ConvertLayer.normalize_image(img_data,
                                                        value_range=self.step_metadata['normalize'],
                                                        logger=self.logger,
                                                        raise_error=self.raise_error)
            if self.step_metadata.get('resize'):
                img_data = ConvertLayer.resize(img_data,
                                               value_range=self.step_metadata['resize'],
                                               logger=self.logger,
                                               raise_error=self.raise_error)
            if self.step_metadata.get('crop'):
                img_data = ConvertLayer.crop_images(img_data,
                                                   value_range=self.step_metadata['crop'],
                                                   logger=self.logger,
                                                   raise_error=self.raise_error)

            img = ImageLayer(image=img_data,
                            image_metadata=self.get_catalog(flag='all').iloc[index].to_dict(),
                            step_order=self.images.get_last_step_order(),
                            step_name=self.step_name)
            image_list.append(img)
        if not test:
            self.step_name = f'convert_{self.config_index}' if not self.step_name else self.step_name
            self.images.add_step(step=self.step_name,
                                 images=image_list,
                                 step_metadata=self.step_metadata)

    @staticmethod
    def convert_bits(img, output_bits = None, autoscale = None, logger = None, raise_error = False):
        if autoscale:
            try:
                result = np.float32(img) - np.min(img)
                result[result < 0.0] = 0.0
                if np.max(img) != 0:
                    result = result / np.max(img)
                if output_bits == 8:
                    img_bit = np.uint8(255 * result)
                elif output_bits == 16:
                    img_bit = np.uint16(65535 * result)
                elif output_bits == 32:
                    img_bit = np.float32(result)
            except Exception as e:
                if logger:
                    logger.error("Failed to autoscale the image: %s", e)
                if raise_error:
                    raise ValueError(f"Failed to autoscale the image: {str(e)}") from e
                img_bit = img
        else:
            if output_bits == 8:
                img_bit = np.uint8(255)
            elif output_bits == 16:
                img_bit = np.uint16(65535)
            elif output_bits == 32:
                img_bit = np.float32(result)
        return img_bit

    @staticmethod
    def channel_convert(img, to = None, channel_selector = None, logger = None, raise_error = False):
        try:
            if to == 'RGB':
                if len(img.shape) != 3 and img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif to == 'gray':
                if channel_selector in [0, 1, 2]:
                    img = img[:, :, channel_selector]
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            if logger:
                logger.error(f"Failed to convert the image to {to}: {str(e)}")
            if raise_error:
                raise ValueError(f"Failed to convert the image to {to}: {str(e)}") from e
        return img

    @staticmethod
    def get_bayer_pattern(img, bayer_pattern, logger=None, raise_error=False):
        # Determine the number of channels in the input image
        if len(img.shape) == 3:
            img_channels = img.shape[2]
        else:
            img_channels = 1
        
        if img_channels == 1:
            if bayer_pattern == 'RG':
                return cv2.COLOR_BAYER_RG2RGB
            elif bayer_pattern == 'BG':
                return cv2.COLOR_BAYER_BG2RGB
            elif bayer_pattern == 'GR':
                return cv2.COLOR_BAYER_GR2RGB
            elif bayer_pattern == 'GB':
                return cv2.COLOR_BAYER_GB2RGB
            else:
                if logger:
                    logger.warning("Invalid Bayer pattern for a single-channel image: %s", bayer_pattern)
                if raise_error:
                    raise ValueError("Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'.")
                return None
        elif img_channels in [3, 4]:
            if logger:
                logger.warning("Unsupported number of channels in the image: %s", img_channels)
            if raise_error:
                raise ValueError("Invalid Bayer pattern for a single-channel image. Expected 'RG', 'BG', 'GR', or 'GB'.")
            return None
        else:
            if logger:
                logger.warning("Unsupported number of channels in the image: %s", img_channels)
            if raise_error:
                raise ValueError("Unsupported number of channels in the image.")
            return None


    @staticmethod
    def normalize_image(img, value_range=None, logger=None, raise_error=False):
        try:
            return cv2.normalize(img, img, value_range[0], value_range[1], cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        except Exception as e:
            if logger:
                logger.error(f"Failed to normalize the image: {str(e)}")
            if raise_error:
                raise ValueError(f"Failed to normalize the image: {str(e)}") from e
        return img

    @staticmethod
    def resize(img, value_range, logger=None, raise_error=False):
        try:
            return cv2.resize(img, (value_range[0], value_range[1]), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            if logger:
                logger.error(f"Failed to resize the image: {str(e)}")
            if raise_error:
                raise ValueError(f"Failed to resize the image: {str(e)}") from e
        return img

    @staticmethod
    def crop_images(img, value_range, logger=None, raise_error=False):
        try:
            start_x, end_x = value_range[0]
            start_y, end_y = value_range[1]
            if start_x < 0 or end_x > img.shape[0] or start_y < 0 or end_y > img.shape[1]:
                raise ValueError("Crop range is out of bounds.")
            return img[start_x:end_x, start_y:end_y, :]
        except Exception as e:
            if logger:
                logger.error("Failed to crop the image: %s", str(e))
            if raise_error:
                raise ValueError(f"Failed to crop the image: {str(e)}") from e
        return img
