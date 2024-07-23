import sys
import yaml
from pathlib import Path


class GeneralConfig:
    def __init__(self,
                 input_path,
                 output_path,
                 image_type,
                 catalog_path=None,
                 catalog_type=None,
                 file_name_pattern=None,
                 channels=None,
                 autoscale=None,
                 bayer_pattern=None):

        self.input_path = input_path
        self.output_path = output_path
        self.catalog_path = catalog_path
        self.catalog_type = catalog_type
        self.image_type = image_type
        self.file_name_pattern = file_name_pattern
        self.channels = channels
        self.autoscale = autoscale
        self.bayer_pattern = bayer_pattern


class PositionConfig:
    def __init__(self,
                 waypoint_format,
                 waypoint_folder_path,
                 waypoint_data_pattern):

        self.waypoint_format = waypoint_format
        self.waypoint_folder_path = waypoint_folder_path
        self.waypoint_data_pattern = waypoint_data_pattern

class SamplingConfig:
    def __init__(self,
                 path_input,
                 path_output,
                 image_type,
                 sampling_method,
                 sampling_percent,
                 target_size,
                 cutting_ruler,
                 invert_img):
        self.path_input = path_input
        self.path_output = path_output
        self.image_type = image_type
        self.sampling_method = sampling_method
        self.sampling_percent = sampling_percent
        self.target_size = target_size
        self.cutting_ruler = cutting_ruler
        self.invert_img = invert_img


class CVAttrConfig:
    def __init__(self,
                 name,
                 version,
                 description,
                 file_name_fmt,
                 edge_threshold_low,
                 edge_threshold_high,
                 edge_detector,
                 downsample_factor,
                 channel_selector,
                 object_selection,
                 is_raw,
                 is_bayer_pattern,
                 autoscale,
                 deconv,
                 deconv_method,
                 deconv_iter,
                 deconv_mask_weight,
                 estimate_sharpness,
                 small_float_val,
                 bw_blur_radius,
                 bayer_pattern,
                 channels):
        self.name = name
        self.version = version
        self.description = description
        self.file_name_fmt = file_name_fmt
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.edge_detector = edge_detector
        self.downsample_factor = downsample_factor
        self.channel_selector = channel_selector
        self.object_selection = object_selection
        self.is_raw = is_raw
        self.is_bayer_pattern = is_bayer_pattern
        self.autoscale = autoscale
        self.deconv = deconv
        self.deconv_method = deconv_method
        self.deconv_iter = deconv_iter
        self.deconv_mask_weight = deconv_mask_weight
        self.estimate_sharpness = estimate_sharpness
        self.small_float_val = small_float_val
        self.bw_blur_radius = bw_blur_radius
        self.bayer_pattern = bayer_pattern
        self.channels = channels


class PreprocessingConfig:
    def __init__(self, denoise):
        self.denoise = denoise


class Configuration:
    def __init__(self, config_file_path):
        try:
            with open(config_file_path, "r", encoding='utf-8') as config_file:
                config_data = yaml.safe_load(config_file)
        except FileNotFoundError as e:
            print(f"Failed to load the configuration file: {str(e)}")
            raise FileNotFoundError(f"Failed to load the configuration file: {str(e)}") from e
        except yaml.YAMLError as e:
            print(f"Failed to load the configuration file: {str(e)}")
            raise yaml.YAMLError(f"Failed to load the configuration file: {str(e)}") from e

        if 'general' not in config_data:
            print("General configuration is not specified.")
            sys.exit()
        self.general = GeneralConfig(**config_data['general'])

        if 'position' in config_data:
            self.position = PositionConfig(**config_data['position'])

        if 'sampling' in config_data:
            self.sampling = SamplingConfig(**config_data['sampling'])

        if 'cv_attribute' in config_data:
            self.cv_attribute = CVAttrConfig(**config_data['cv_attribute'])

        if 'preprocessing' in config_data:
            self.preprocessing = PreprocessingConfig(**config_data['preprocessing'])

    def write(self, filename):
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with filename.open("w", encoding='utf-8') as file_handler:
            yaml.dump(
                self, file_handler, allow_unicode=True, default_flow_style=False
            )
