import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import mariqt.sources.ifdo as miqtifdo
import mariqt.tests as miqtt
from mariqt.core import IfdoException

class CatalogParser:
    def __init__(self, config, catalog_path = None, catalog_type = None):
        self.config = config
        self.catalog_type = catalog_type if catalog_type else self.config.general.catalog_type
        if not self.catalog_type:
            print("Catalog type is not specified.")
            raise ValueError("Catalog type is not specified.")

        catalog_path = catalog_path if catalog_path else self.config.general.catalog_path
        if not catalog_path:
            print("Catalog path is not specified.")
            raise ValueError("Catalog path is not specified.")
        self.catalog_path = Path(catalog_path)
        self.catalog = self.open_catalog()

    def open_catalog(self):
        if self.catalog_type == 'IFDO':
            catalog = self.open_ifdo_catalog()
        elif self.catalog_type == 'CSV':
            catalog = self.open_csv_catalog()
        else:
            print("Catalog type is not supported.")
            raise ValueError("Catalog type is not supported.")
        return catalog

    def open_ifdo_catalog(self):
        catalog = miqtifdo.iFDO_Reader(self.catalog_path).ifdo
        CatalogParser.validate_ifdo(catalog)
        return catalog

    def open_csv_catalog(self):
        filename_columns = ['filename', 'file_name', 'FileName', 'File Name']
        index_columns = ['id', 'index', 'ID', 'Index', 'Id']
        catalog = pd.read_csv(self.catalog_path)
        if not any(col in catalog.columns for col in filename_columns):
            raise ValueError("Catalog does not have a filename column. It should have one of the following columns: 'filename', 'file_name', 'FileName', 'File Name'.")
        if not any(col in catalog.columns for col in index_columns):
            catalog = catalog.reset_index().rename(columns={'index': 'ID'})
        for col in filename_columns:
            if col in catalog.columns:
                catalog.rename(columns={col: 'filename'}, inplace=True)
                break
        filename_columns.remove(col)
        catalog.drop(filename_columns, errors='ignore', inplace=True)
        return catalog

    @staticmethod
    def validate_ifdo(ifdo_data: Dict):
        """Validates input data against iFDO scheme. Found errors are printed to the terminal.

        Args:
            ifdo_data (Dict): parsed iFDO data.
        """ 
        miqtt.are_valid_ifdo_fields(ifdo_data["image-set-header"])
        unique_names = miqtt.filesHaveUniqueName(ifdo_data['image-set-items'].keys())
        if unique_names is not None:
            print(f"Duplicate filenames found: {unique_names}")
            raise IfdoException({"Validation error": f"Duplicate filenames found: {unique_names}"})
