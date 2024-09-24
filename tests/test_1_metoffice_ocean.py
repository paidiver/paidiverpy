""" Tests for the MetOfficeOcean class.
"""

import unittest

from unittest.mock import patch
from datetime import datetime
import os
import glob
from dotenv import load_dotenv
import xarray as xr

from tests.base_test_class import BaseTestClass

from .mocks.mock_ftp import MockFTP

from metoffice_data_handler.utils import remove_files
from metoffice_data_handler.geoserver_connection import GeoserverConnection

with patch("ftplib.FTP", new=MockFTP):
    from metoffice_data_handler.metoffice_ocean import MetOfficeOcean

load_dotenv()


class TestMetOfficeOcean(BaseTestClass):
    """Tests for the MetOfficeOcean class.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    @patch("metoffice_data_handler.metoffice_ocean.FTP", new=MockFTP)
    def test_download_files(self):
        """Test the download_files method."""

        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15", "shelf-amm7", "global-ocean"],
            model_date=datetime.now().strftime("%Y%m%d"),
            output_type="hi",
            forecast_days=[-2, -1, 0, 1, 2, 3, 4, 5, 6],
            variables=["TEM"],
            worker_threads=1,
            logger=self.logger,
        )
        metoffice_ocean.download_files()
        downloaded_files = glob.glob(os.path.join(self.test_dir, "*TEM*.nc"))
        self.assertGreater(len(downloaded_files), 0, "No files were downloaded.")
        self.assertEqual(len(downloaded_files), 27)

    def test_download_files_cur(self):
        """Test the download_files method."""
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date=datetime.now().strftime("%Y%m%d"),
            output_type="hi",
            forecast_days=[-2, -1, 0, 1, 2, 3, 4, 5, 6],
            variables=["CUR"],
            worker_threads=4,
            logger=self.logger,
        )
        metoffice_ocean.download_files()
        downloaded_files = glob.glob(os.path.join(self.test_dir, "*CUR*.nc"))
        self.assertGreater(len(downloaded_files), 0, "No files were downloaded.")
        self.assertEqual(len(downloaded_files), 9)

    def test_process_files_tem(self):
        """Test the process_files method."""
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date="20240819",
            output_type="hi",
            forecast_days=[1, 2],
            variables=["TEM"],
            worker_threads=4,
            logger=self.logger,
        )
        metoffice_ocean.process_files(local_path=str(self.local_path))
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*TEM*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 2)

        ds = xr.load_dataset(
            os.path.join(
                self.local_path,
                "output_ocean_shelf-amm15_run20240819_forecast1_TEM_hi.nc",
            )
        )
        self.assertEqual(
            dict(ds.sizes), {"time": 2, "latitude": 1240, "longitude": 958, "z": 2}
        )
        self.assertEqual(len(ds.coords), 4)
        self.assertEqual(list(ds.coords), ["time", "latitude", "longitude", "z"])
        self.assertEqual(len(ds.data_vars), 1)
        self.assertEqual(list(ds.data_vars), ["thetao"])
        remove_files(self.local_path, "ocean", logger=self.logger, output=True)
        converted_files = glob.glob(os.path.join(self.local_path, "output*ocean*.nc"))
        self.assertEqual(len(converted_files), 0)

    def test_process_files_tem2(self):
        """Test the process_files method."""
        model_date = "latest"
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date=model_date,
            output_type="hi",
            forecast_days=[1, 2],
            variables=["TEM"],
            worker_threads=4,
            logger=self.logger,
        )
        metoffice_ocean.process_files(local_path=str(self.local_path))
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*TEM*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 2)

        ds = xr.load_dataset(
            os.path.join(
                self.local_path,
                "output_ocean_shelf-amm15_run20240819_forecast1_TEM_hi.nc",
            )
        )
        self.assertEqual(
            dict(ds.sizes), {"time": 2, "latitude": 1240, "longitude": 958, "z": 2}
        )
        self.assertEqual(len(ds.coords), 4)
        self.assertEqual(list(ds.coords), ["time", "latitude", "longitude", "z"])
        self.assertEqual(len(ds.data_vars), 1)
        self.assertEqual(list(ds.data_vars), ["thetao"])
        remove_files(self.local_path, "ocean", logger=self.logger, output=True)
        converted_files = glob.glob(os.path.join(self.local_path, "output*ocean*.nc"))
        self.assertEqual(len(converted_files), 0)

    def test_process_files_tem_error(self):
        """Test the process_files method."""
        model_date = "20240820"

        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date=model_date,
            output_type="hi",
            forecast_days=[1, 2],
            variables=["TEM"],
            worker_threads=4,
            logger=self.logger,
        )

        with self.assertRaises(KeyError) as cm:
            metoffice_ocean.process_files(
                local_path=str(self.local_path), model_date=model_date
            )
        self.assertEqual(str(cm.exception), "'No files found for model run 20240820'")

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*TEM*.nc")
        )
        self.assertEqual(len(converted_files), 0)

    def test_process_files_cur(self):
        """Test the process_files method."""
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date="20240819",
            output_type="hi",
            forecast_days=[1, 2],
            variables=["CUR"],
            worker_threads=4,
            logger=self.logger,
        )

        metoffice_ocean.process_files(local_path=str(self.local_path))

        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*CUR*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 2)
        ds = xr.load_dataset(
            os.path.join(
                self.local_path,
                "output_ocean_shelf-amm15_run20240819_forecast1_CUR_hi.nc",
            )
        )
        self.assertEqual(
            dict(ds.sizes), {"time": 2, "latitude": 1240, "longitude": 958, "z": 2}
        )
        self.assertEqual(len(ds.coords), 4)
        self.assertEqual(list(ds.coords), ["time", "latitude", "longitude", "z"])
        self.assertEqual(len(ds.data_vars), 3)
        self.assertEqual(list(ds.data_vars), ["uo", "vo", "speed"])
        remove_files(self.local_path, "ocean", logger=self.logger, output=True)
        converted_files = glob.glob(os.path.join(self.local_path, "output*ocean*.nc"))
        self.assertEqual(len(converted_files), 0)

    def test_upload_geoserver_tem(self):
        """Test the upload_to_geoserver method."""
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date="20240819",
            output_type="hi",
            forecast_days=[1, 2],
            variables=["TEM"],
            worker_threads=4,
            logger=self.logger,
        )
        metoffice_ocean.process_files(local_path=str(self.local_path))
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*TEM*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 2)

        geo = GeoserverConnection()
        workspace = geo.get_workspace("metoffice_ocean")
        if workspace is not None:
            geo.remove_workspace("metoffice_ocean", recurse=True)
        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM1")
        self.assertIsNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM2")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM1", "thetao1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM2", "thetao2")
        self.assertIsNone(coverage_layer)

        metoffice_ocean.upload_to_geoserver()

        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNotNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM1")
        self.assertIsNotNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM2")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM1", "thetao1")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM2", "thetao2")
        self.assertIsNotNone(coverage_layer)

        geo.remove_workspace("metoffice_ocean", recurse=True)
        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM1")
        self.assertIsNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM2")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM1", "thetao1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM2", "thetao2")
        self.assertIsNone(coverage_layer)
        remove_files(self.local_path, "ocean", logger=self.logger, output=True)
        converted_files = glob.glob(os.path.join(self.local_path, "output*ocean*.nc"))
        self.assertEqual(len(converted_files), 0)

    def test_upload_geoserver_cur(self):
        """Test the upload_to_geoserver method."""
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date="20240819",
            output_type="hi",
            forecast_days=[1, 2],
            variables=["CUR"],
            worker_threads=4,
            logger=self.logger,
        )
        metoffice_ocean.process_files(local_path=str(self.local_path))
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*CUR*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 2)

        geo = GeoserverConnection()
        workspace = geo.get_workspace("metoffice_ocean")
        if workspace is not None:
            geo.remove_workspace("metoffice_ocean", recurse=True)
        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "CUR1")
        self.assertIsNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "CUR2")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "uo1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "uo2")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "vo1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "vo2")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "speed1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "speed2")
        self.assertIsNone(coverage_layer)

        metoffice_ocean.upload_to_geoserver()

        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNotNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "CUR1")
        self.assertIsNotNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "CUR2")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "uo1")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "uo2")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "vo1")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "vo2")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "speed1")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "speed2")
        self.assertIsNotNone(coverage_layer)

        geo.remove_workspace("metoffice_ocean", recurse=True)
        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "CUR1")
        self.assertIsNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "CUR2")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "uo1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "uo2")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "vo1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "vo2")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR1", "speed1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "CUR2", "speed2")
        self.assertIsNone(coverage_layer)

        remove_files(self.local_path, "ocean", logger=self.logger, output=True)
        converted_files = glob.glob(os.path.join(self.local_path, "output*ocean*.nc"))
        self.assertEqual(len(converted_files), 0)

    def test_upload_geoserver_tem2(self):
        """Test the upload_to_geoserver method."""
        metoffice_ocean = MetOfficeOcean(
            local_path=str(self.test_dir),
            models=["shelf-amm15"],
            model_date="20240819",
            output_type="hi",
            forecast_days=[1, 2],
            variables=["TEM"],
            worker_threads=4,
            logger=self.logger,
        )
        metoffice_ocean.process_files(local_path=str(self.local_path))
        converted_files = glob.glob(
            os.path.join(self.local_path, "output*ocean*TEM*.nc")
        )
        self.assertGreater(len(converted_files), 0, "No files were downloaded.")
        self.assertEqual(len(converted_files), 2)

        geo = GeoserverConnection()
        workspace = geo.get_workspace("metoffice_ocean")
        if workspace is not None:
            geo.remove_workspace("metoffice_ocean", recurse=True)
        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM1")
        self.assertIsNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM2")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM1", "thetao1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM2", "thetao2")
        self.assertIsNone(coverage_layer)

        metoffice_ocean.upload_to_geoserver(styles=["viridis", "jet"])
        metoffice_ocean.upload_to_geoserver(styles=["viridis", "jet"])

        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNotNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM1")
        self.assertIsNotNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM2")
        self.assertIsNotNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM1", "thetao1")
        self.assertIsNotNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM2", "thetao2")
        self.assertIsNotNone(coverage_layer)

        geo.create_layer_style(
            workspace_name="metoffice_ocean",
            layer_name="thetao2",
            style_params={
                "colormap_name": ["plasma"],
            },
            file_path=os.path.join(
                self.local_path,
                "output_ocean_shelf-amm15_run20240819_forecast2_TEM_hi.nc",
            ),
            variable="thetao",
        )
        geo.remove_workspace("metoffice_ocean", recurse=True)
        workspace = geo.get_workspace("metoffice_ocean")
        self.assertIsNone(workspace)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM1")
        self.assertIsNone(coverage_store)
        coverage_store = geo.get_coverage_store("metoffice_ocean", "TEM2")
        self.assertIsNone(coverage_store)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM1", "thetao1")
        self.assertIsNone(coverage_layer)
        coverage_layer = geo.get_coverage_layer("metoffice_ocean", "TEM2", "thetao2")
        self.assertIsNone(coverage_layer)
        remove_files(self.local_path, "ocean", logger=self.logger, output=True)
        converted_files = glob.glob(os.path.join(self.local_path, "output*ocean*.nc"))
        self.assertEqual(len(converted_files), 0)


if __name__ == "__main__":
    unittest.main()
