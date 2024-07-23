""" Open raw image file
"""
import os
import shutil
from datetime import datetime
import glob
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from paidiverpy import Paidiverpy
from .resampling_tools import (
    sampling_percent,
    sampling_fixed_number,
    load_uvp5,
    copy_image_from_df
)

class ResampleLayer(Paidiverpy):
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None):
        super().__init__(config_file_path, input_path, output_path, catalog_path, catalog_type, catalog, config)
        self.waypoint_data = []

    def resampling(self, save_output=False):
        output_folder = Path(self.config.sampling.path_output)

        if self.config.sampling.image_type == 'UVP5':
            # load uvp5
            df = load_uvp5(self.config.sampling.path_input)
        else:
            raise ValueError("Image type not supported")

        if self.config.sampling.sampling_method == 'fixed':
            df_sample = sampling_fixed_number(df, self.config.sampling.sampling_percent)
        elif self.config.sampling.sampling_method == 'percent':
            df_sample = sampling_percent(df, self.config.sampling.sampling_percent)
        else:
            raise ValueError("Sampling method not supported")

        # create sampling output
        time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        sampling_path = output_folder / ("sampling" + time_str)
        # if not sampling_path_train.exists():
        sampling_path.mkdir(parents=True, exist_ok=True)
        sampling_path_images = sampling_path / "output"
        sampling_path_images.mkdir(parents=True, exist_ok=True)

        # Loop through the image paths and copy the images to the target directory
        if df_sample is not None:
            df_sample = copy_image_from_df(df_sample, sampling_path_images, self.config.sampling.target_size,
                                        cutting_ruler=self.config.sampling.cutting_ruler,
                                        invert_img=self.config.sampling.invert_img)

        # shuffle and remove redundant columns
        df_final = df_sample.drop('path', axis=1)
        df_final = df_final.sample(frac=1, random_state=np.random.seed())
        df_final = df_final.loc[:, ~df_final.columns.str.contains('^Unnamed')]
        df_final = df_final.reset_index()
        df_final = df_final.replace('NaN', 0)

        csv_path = sampling_path / "sampled_images.csv"
        df_final.to_csv(csv_path)

    def remove_obscure_photos(save_output=False):
        pass