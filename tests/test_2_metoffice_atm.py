""" Tests for the MetOfficeAtm class.
"""

import os
import unittest
import warnings
import glob
from unittest.mock import patch

import xarray as xr
from dotenv import load_dotenv

from tests.base_test_class import BaseTestClass
from metoffice_data_handler.geoserver_connection import GeoserverConnection
from metoffice_data_handler.metoffice_atm import MetOfficeAtm
from metoffice_data_handler.utils import initialise_logging, remove_files

from .mocks import mock_requests

load_dotenv()

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestMetOfficeAtm(BaseTestClass):
    """Tests for the MetOfficeAtm class.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_error_init_class(self):
        """Test the error when the class is initialised without the required parameters."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        self.tearDownClass()

        with self.assertRaises(ValueError) as cm:
            MetOfficeAtm(
                orders_id=[],
                worker_threads=2,
            )
        self.assertEqual(str(cm.exception), "Orders ID must be supplied.")

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_download_files(self):
        """Test the download_files method."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        self.tearDownClass()
        metoffice_atm = MetOfficeAtm(
            orders_id=["123456"],
            worker_threads=2,
        )
        metoffice_atm.download_files(
            local_path=str(self.test_dir),
            model_run="latest",
        )
        downloaded_files = glob.glob(
            os.path.join(self.test_dir, "atmosphere*agl*.grib2")
        )
        self.assertGreater(len(downloaded_files), 0, "No files were downloaded.")
        self.assertEqual(len(downloaded_files), 2)

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_download_files_error_order(self):
        """Test the download_files method with an error in the order."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        self.tearDownClass()
        metoffice_atm = MetOfficeAtm(
            orders_id=["654321"],
            worker_threads=2,
        )
        with self.assertRaises(ValueError) as cm:
            metoffice_atm.download_files(
                local_path=str(self.test_dir),
            )
        self.assertEqual(
            str(cm.exception),
            "No models could be extracted from the orders to process.",
        )

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_download_error_model_date(self):
        """Test the download_files method with an error in the model date."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        os.environ["METOFFICE_DATAHUB_DEFAULT_ORDER"] = "123456"

        self.tearDownClass()

        metoffice_atm = MetOfficeAtm(
            worker_threads=2,
        )
        metoffice_atm.download_files(
            local_path=str(self.test_dir),
            model_date="20220101",
        )
        downloaded_files = glob.glob(
            os.path.join(self.test_dir, "atmosphere*agl*.grib2")
        )
        self.assertEqual(len(downloaded_files), 0)

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_download_error_model_run(self):
        """Test the download_files method with an error in the model run."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        self.tearDownClass()
        metoffice_atm = MetOfficeAtm(
            orders_id=["123456"],
            worker_threads=2,
        )
        with self.assertRaises(ValueError) as cm:
            metoffice_atm.download_files(
                local_path=str(self.test_dir),
                model_run=["07"],
            )
        self.assertEqual(
            str(cm.exception), "No model runs available with the selected runs"
        )

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_process_files(self):
        """Test the process_files method."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        metoffice_atm = MetOfficeAtm(
            orders_id=["123456"],
            worker_threads=2,
        )
        metoffice_atm.process_files(
            local_path=str(self.local_path), model_date="20240819T00"
        )

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        ds = xr.load_dataset(
            os.path.join(self.local_path, "output_atmosphere_20240819T00.nc")
        )
        self.assertEqual(
            dict(ds.sizes), {"time": 25, "latitude": 612, "longitude": 482}
        )
        self.assertEqual(len(ds.coords), 3)
        self.assertEqual(list(ds.coords), ["latitude", "longitude", "time"])
        self.assertEqual(len(ds.data_vars), 3)
        self.assertEqual(list(ds.data_vars), ["uo", "vo", "wind_speed"])
        remove_files(self.local_path, "atmosphere", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_process_files_error(self):
        """Test the process_files method with an error."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        metoffice_atm = MetOfficeAtm(
            orders_id=["123456"],
            worker_threads=2,
        )
        with self.assertRaises(KeyError) as cm:
            metoffice_atm.process_files(
                local_path=str(self.local_path), model_date="20240818"
            )
        self.assertEqual(str(cm.exception), "'No files found for model run 20240818'")

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)
        remove_files(self.local_path, "atmosphere", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_process_files_error2(self):
        """Test the process_files method with an error."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        metoffice_atm = MetOfficeAtm(
            orders_id=["123456"],
            worker_threads=2,
        )
        with self.assertRaises(KeyError) as cm:
            new_path = os.path.join(self.local_path, "files_error")
            metoffice_atm.process_files(local_path=new_path)
        self.assertEqual(str(cm.exception), "'No files found for latest model run'")

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)
        remove_files(self.local_path, "atmosphere", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def init_metoffice_atm(self):
        """Initialise the MetOfficeAtm class."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        logger = initialise_logging()

        metoffice_atm = MetOfficeAtm(
            orders_id=["123456"], worker_threads=2, logger=logger
        )
        metoffice_atm.process_files(
            local_path=str(self.local_path)  # , model_date="20240819T00"
        )
        return metoffice_atm

    def test_upload_geoserver(self):
        """Test the upload_to_geoserver method."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        metoffice_atm = self.init_metoffice_atm()

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        geo = GeoserverConnection()
        workspace = geo.get_workspace("metoffice_atm")
        if workspace is not None:
            geo.remove_workspace("metoffice_atm", recurse=True)
        workspace = geo.get_workspace("metoffice_atm")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_atm", "wind")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "uo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "vo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "wind_speed")
        self.assertIsNone(coverage_layer)

        metoffice_atm.upload_to_geoserver(
            output_filename="output_atmosphere_20240819T00.nc",
            local_path=self.local_path,
        )

        workspace = geo.get_workspace("metoffice_atm")
        self.assertIsNotNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_atm", "wind")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "uo")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "vo")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "wind_speed")
        self.assertIsNotNone(coverage_layer)

        geo.remove_workspace("metoffice_atm", recurse=True)
        workspace = geo.get_workspace("metoffice_atm")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_atm", "wind")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "uo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "vo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "wind_speed")
        self.assertIsNone(coverage_layer)

        remove_files(self.local_path, "atmosphere", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)

    def test_upload_geoserver2(self):
        """Test the upload_to_geoserver method."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "123456"
        metoffice_atm = self.init_metoffice_atm()

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        geo = GeoserverConnection()
        workspace = geo.get_workspace("metoffice_atm")
        if workspace is not None:
            geo.remove_workspace("metoffice_atm", recurse=True)
        workspace = geo.get_workspace("metoffice_atm")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_atm", "wind")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "uo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "vo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "wind_speed")
        self.assertIsNone(coverage_layer)

        metoffice_atm.upload_to_geoserver()

        workspace = geo.get_workspace("metoffice_atm")
        self.assertIsNotNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_atm", "wind")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "uo")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "vo")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "wind_speed")
        self.assertIsNotNone(coverage_layer)

        geo.remove_workspace("metoffice_atm", recurse=True)
        workspace = geo.get_workspace("metoffice_atm")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_atm", "wind")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "uo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "vo")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_atm", "wind", "wind_speed")
        self.assertIsNone(coverage_layer)
        # try remove files with error
        remove_files(self.local_path, "atmosphre", output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        # try remove files with error
        remove_files(None, None, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 1)

        remove_files(self.local_path, "atmosphere", logger=self.logger, output=True)
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*atmosphere*.nc")
        )
        self.assertEqual(len(converted_files), 0)

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_error_init_class2(self):
        """Test the error when the class is initialised without the required parameters."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = ""
        self.tearDownClass()

        with self.assertRaises(ValueError) as cm:
            MetOfficeAtm(
                orders_id=["123456"],
                worker_threads=2,
            )
        self.assertEqual(str(cm.exception), "API credentials must be supplied.")

    @patch("metoffice_data_handler.metoffice_atm.requests.get", new=mock_requests.get)
    def test_error_zero_orders(self):
        """Test the error when the class is initialised without the required parameters."""
        os.environ["METOFFICE_DATAHUB_API_KEY"] = "654321"
        self.tearDownClass()

        with self.assertRaises(ValueError) as cm:
            MetOfficeAtm(
                orders_id=["123456"],
                worker_threads=2,
            )
        self.assertEqual(str(cm.exception), "No orders active on Weather DataHub.")


if __name__ == "__main__":
    unittest.main()
