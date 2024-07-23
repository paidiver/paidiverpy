import glob
import os
from pathlib import Path

import pandas as pd
from paidiverpy.catalog_parser import CatalogParser
from paidiverpy.config import Configuration

class Paidiverpy:
    def __init__(self,
                 config_file_path=None,
                 input_path=None,
                 output_path=None,
                 catalog_path=None,
                 catalog_type=None,
                 catalog=None,
                 config=None):
        self.config = config
        if not config:
            self.config = Configuration(config_file_path)
        if not config_file_path:
            self.input_path = Path(input_path)
            self.output_path = Path(output_path)
        else:
            self.input_path = Path(self.config.general.input_path)
            self.output_path = Path(self.config.general.output_path)
        self.catalog = catalog if catalog is not None else self.open_catalog(catalog_path=catalog_path, catalog_type=catalog_type)

    def open_catalog(self, catalog_path = None, catalog_type = None):

        if self.config.general.catalog_type:
            catalog_parser = CatalogParser(self.config, catalog_path, catalog_type)
            catalog = catalog_parser.open_catalog()
        else:
            file_pattern = str(self.input_path.joinpath(self.config.general.file_name_pattern))
            list_of_files = glob.glob(file_pattern)
            list_of_files = [os.path.basename(file) for file in list_of_files]
            catalog = pd.DataFrame(list_of_files, columns=['filename'])
            catalog = catalog.reset_index().rename(columns={'index': 'ID'})
        return catalog
