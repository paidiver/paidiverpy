""" Tests for the MetOfficeInnovations class.
"""

import unittest

import os
import glob
from dotenv import load_dotenv
import geopandas as gpd
import xarray as xr

from tests.base_test_class import BaseTestClass
from metoffice_data_handler.geoserver_connection import GeoserverConnection
from metoffice_data_handler.utils import remove_files

from metoffice_data_handler.metoffice_innovations import MetOfficeInnovations

load_dotenv()


class TestMetOfficeInnovations(BaseTestClass):
    """Tests for the MetOfficeInnovations class.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_download_files(self):
        """Test the download_files method."""

        metoffice_innovations = MetOfficeInnovations(
            worker_threads=2,
            local_path=str(self.test_dir),
        )
        metoffice_innovations.download_files()
        downloaded_files = glob.glob(os.path.join(self.test_dir, "innovations*"))
        self.assertGreater(len(downloaded_files), 0, "No files were downloaded.")
        self.assertEqual(len(downloaded_files), 2)
        remove_files(self.local_path, "innovations", logger=self.logger, output=True)

    def test_process_files(self):
        """Test the process_files method."""
        metoffice_innovations = MetOfficeInnovations(
            worker_threads=2,
            local_path=str(self.local_path),
        )
        metoffice_innovations.process_files(
            local_path=str(self.local_path), variables=["TEM"]
        )
        converted_files = glob.glob(
            os.path.join(
                self.local_path, "output_innovations_20240819_increments_TEM.nc"
            )
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        ds = xr.load_dataset(
            os.path.join(
                self.local_path, "output_innovations_20240819_increments_TEM.nc"
            )
        )
        self.assertEqual(dict(ds.sizes), {"z": 2, "latitude": 1345, "longitude": 1458})
        self.assertEqual(len(ds.coords), 3)
        self.assertEqual(list(ds.coords), ["z", "latitude", "longitude"])
        self.assertEqual(len(ds.data_vars), 1)
        self.assertEqual(list(ds.data_vars), ["TEM"])

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*innovations*model-errors*")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 5)
        gdf = gpd.read_file(
            os.path.join(
                self.local_path, "output_innovations_20240819_model-errors_.shp"
            )
        )
        self.assertEqual(len(gdf.columns), 2)
        self.assertEqual(list(gdf.columns), ["value", "geometry"])
        self.assertEqual(gdf.shape, (89, 2))
        remove_files(self.local_path, "innovations", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*innovations*")
        )
        self.assertEqual(len(converted_files), 0)

    def test_upload_geoserver(self):
        """Test the upload_to_geoserver method."""

        metoffice_innovations = MetOfficeInnovations(
            worker_threads=2,
            local_path=str(self.local_path),
        )
        metoffice_innovations.process_files(
            local_path=str(self.local_path), variables=["TEM"]
        )

        converted_files = glob.glob(
            os.path.join(
                self.local_path, "output_innovations_20240819_increments_TEM.nc"
            )
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*innovations*model-errors*")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 5)

        geo = GeoserverConnection()
        workspace = geo.get_workspace("metoffice_innovations")
        if workspace is not None:
            geo.remove_workspace("metoffice_innovations", recurse=True)
        workspace = geo.get_workspace("metoffice_innovations")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store(
            "metoffice_innovations", "innovations_TEM"
        )
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer(
            "metoffice_innovations", "innovations_TEM", "TEM"
        )
        self.assertIsNone(coverage_layer)

        data_store = geo.get_data_store(
            "metoffice_innovations", "innovations_model-errors"
        )
        self.assertIsNone(data_store)
        data_layer = geo.get_data_layer(
            "metoffice_innovations", "innovations_model-errors", "model_errors"
        )
        self.assertIsNone(data_layer)

        metoffice_innovations.upload_to_geoserver()

        workspace = geo.get_workspace("metoffice_innovations")
        self.assertIsNotNone(workspace)
        coverage_store = geo.get_coverage_store(
            "metoffice_innovations", "innovations_TEM"
        )
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer(
            "metoffice_innovations", "innovations_TEM", "TEM"
        )
        self.assertIsNotNone(coverage_store)
        data_store = geo.get_data_store(
            "metoffice_innovations", "innovations_model-errors"
        )
        self.assertIsNotNone(data_store)
        data_layer = geo.get_data_layer(
            "metoffice_innovations", "innovations_model-errors", "model_errors"
        )
        self.assertIsNotNone(data_layer)

        geo.remove_data_layer(
            "metoffice_innovations",
            "innovations_model-errors",
            "model_errors",
            recurse=True,
        )
        geo.remove_data_store("metoffice_innovations", "innovations_model-errors")
        geo.remove_coverage_layer(
            "metoffice_innovations", "innovations_TEM", "TEM", recurse=True
        )
        geo.remove_coverage_store("metoffice_innovations", "innovations_TEM")
        geo.remove_workspace("metoffice_innovations")
        workspace = geo.get_workspace("metoffice_innovations")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store(
            "metoffice_innovations", "innovations_TEM"
        )
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer(
            "metoffice_innovations", "innovations_TEM", "TEM"
        )
        self.assertIsNone(coverage_layer)
        data_store = geo.get_data_store(
            "metoffice_innovations", "innovations_model-errors"
        )
        self.assertIsNone(data_store)
        data_layer = geo.get_data_layer(
            "metoffice_innovations", "innovations_model-errors", "model_errors"
        )
        self.assertIsNone(data_layer)
        remove_files(self.local_path, "innovations", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*innovations*")
        )
        self.assertEqual(len(converted_files), 0)


if __name__ == "__main__":
    unittest.main()
