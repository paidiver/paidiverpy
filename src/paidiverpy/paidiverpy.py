""" Main class for the paidiverpy package.
"""
import glob
import os
from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt
from paidiverpy.catalog_parser import CatalogParser
from paidiverpy.config import Configuration
from paidiverpy.images_layer import ImagesLayer
from utils import initialise_logging

class Paidiverpy:
    """ Main class for the paidiverpy package.

    Args:
        config_file_path (str): The path to the configuration file.
        input_path (str): The path to the input files.
        output_path (str): The path to the output files.
        catalog_path (str): The path to the catalog file.
        catalog_type (str): The type of the catalog file.
        catalog (CatalogParser): The catalog object.
        config (Configuration): The configuration object.
        logger (logging.Logger): The logger object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        raise_error (bool): Whether to raise an error.
        verbose (bool): Whether to print verbose messages.
    """
    def __init__(
        self,
        config_file_path: str=None,
        input_path: str=None,
        output_path: str=None,
        catalog_path: str=None,
        catalog_type: str=None,
        catalog: CatalogParser=None,
        config: Configuration=None,
        logger: logging.Logger=None,
        images: ImagesLayer=None,
        paidiverpy: "Paidiverpy"=None,
        raise_error: bool=False,
        verbose: bool=False,
    ):
        if paidiverpy:
            self.logger = paidiverpy.logger
            self.images = paidiverpy.images
            self.config = paidiverpy.config
            self.catalog = paidiverpy.catalog
            self.verbose = paidiverpy.verbose
            self.raise_error = paidiverpy.raise_error
        else:
            self.logger = logger or initialise_logging(verbose=verbose)
            self.config = config or self._initialize_config(
                config_file_path, input_path, output_path, catalog_path, catalog_type
            )
            self.images = images or ImagesLayer(
                output_path=self.config.general.output_path
            )
            self.catalog = catalog or self._initialize_catalog()
            self.verbose = verbose
            self.raise_error = raise_error

    def _initialize_config(
        self, config_file_path, input_path, output_path, catalog_path, catalog_type
    ):
        
        general_config = {}
        if input_path:
            general_config["input_path"] = input_path
        if output_path:
            general_config["output_path"] = output_path
        if catalog_path:
            general_config["catalog_path"] = catalog_path
        if catalog_type:
            general_config["catalog_type"] = catalog_type

        if config_file_path:
            return Configuration(config_file_path)
        else:
            config = Configuration()
            config.add_config("general", general_config)
            return config

    def _initialize_catalog(self):
        general = self.config.general
        if getattr(general, "catalog_path", None) and getattr(
            general, "catalog_type", None
        ):
            return CatalogParser(config=self.config, logger=self.logger)
        else:
            self.logger.info(
                "Catalog type is not specified. Loading files from the input path."
            )
            self.logger.info(
                "Catalog will be created from the files in the input path."
            )
            file_pattern = str(
                Path(general.input_path).joinpath(general.file_name_pattern)
            )
            list_of_files = glob.glob(file_pattern)
            list_of_files = [os.path.basename(file) for file in list_of_files]
            catalog = pd.DataFrame(list_of_files, columns=["filename"])
            catalog = catalog.reset_index().rename(columns={"index": "ID"})
            return catalog

    def get_catalog(self, flag=None):
        if isinstance(self.catalog, CatalogParser):
            flag = 0 if flag is None else flag
            if flag == "all":
                if "datetime" not in self.catalog.catalog.columns:
                    return self.catalog.catalog
                return self.catalog.catalog.sort_values("datetime")
            if "datetime" not in self.catalog.catalog.columns:
                return self.catalog.catalog[self.catalog.catalog["flag"] <= flag]
            return self.catalog.catalog[
                self.catalog.catalog["flag"] <= flag
            ].sort_values("datetime")
        return self.catalog

    def set_catalog(self, catalog):
        if isinstance(self.catalog, CatalogParser):
            self.catalog.catalog = catalog
        else:
            self.catalog = catalog

    def get_waypoints(self):
        if isinstance(self.catalog, CatalogParser):
            return self.catalog.waypoints
        raise ValueError("Waypoints are not loaded in the catalog.")

    def show_images(self, step_name):
        for image in self.images[step_name]:
            image.show_image()

    def save_images(self, step=None, by_order=False, image_format="png"):
        last = False
        if step is None:
            last = True
        images = self.images.get_step(step, by_order=by_order, last=last)
        self.logger.info("Saving images from step: %s", step if not last else "last")
        output_path = self.config.general.output_path
        for image in images:
            image.save(output_path, image_format=image_format)

    def plot_trimmed_photos(self, new_catalog):
        catalog = self.get_catalog()
        if not "lon" in catalog.columns or not "lon" in new_catalog.columns:
            self.logger.warning(
                "Longitude and Latitude columns are not found in the catalog."
            )
            self.logger.warning("Plotting will not be performed.")
            return
        plt.figure(figsize=(20, 10))
        plt.plot(catalog["lon"], catalog["lat"], ".k")
        plt.plot(new_catalog["lon"], new_catalog["lat"], "or")
        plt.legend(["Original", "After Trim"])
        plt.show()

    def clear_steps(self, value, by_order=True):
        if by_order:
            self.images.remove_steps_by_order(value)
        else:
            self.images.remove_steps_by_name(value)
        catalog = self.get_catalog(flag="all")
        catalog.loc[catalog["flag"] >= value, "flag"] = 0
        self.set_catalog(catalog)

    def _calculate_steps_metadata(self, config_part):
        steps_metadata = {}
        for key, value in config_part.__dict__.items():
            steps_metadata[key] = value
        return steps_metadata
