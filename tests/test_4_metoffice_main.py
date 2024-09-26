""" This module contains the tests for the CLI of the metoffice_data_handler package.
"""

import unittest
from unittest.mock import patch
import argparse
import metoffice_data_handler.cli.main as main_function
from tests.base_test_class import BaseTestClass


class TestMetOfficeDataHandlerCLI(BaseTestClass):
    """Tests for the CLI of the metoffice_data_handler package.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    @patch("metoffice_data_handler.cli.main.remove_files")
    @patch("metoffice_data_handler.cli.main.MetOfficeAtm")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_atm(
        self,
        mock_parse_args: argparse.Namespace,
        mock_metoffice: main_function.MetOfficeAtm,
        mock_remove_files: main_function.remove_files,
    ):
        """Test the cli function for the atmosphere data.

        Args:
            mock_parse_args (argparse.Namespace): The mock for the argparse.Namespace.
            mock_metoffice (main_function.MetOfficeAtm): The mock for the MetOfficeAtm class.
            mock_remove_files (main_function.remove_files): The mock for the remove_files function.
        """
        mock_parse_args.return_value = argparse.Namespace(
            data_type="atm",
            orders_id="order1,order2",
            worker_threads=1,
            model_date="20230101",
            model_run="latest",
            local_path="some/path",
            styles="jet",
            upload=True,
            convert=True,
            remove_files=True,
        )

        main_function.main()

        mock_metoffice.assert_called_once_with(
            orders_id=["order1", "order2"],
            worker_threads=1,
            logger=main_function.logger,
        )
        mock_metoffice.return_value.download_files.assert_called_once_with(
            model_date="20230101", model_run="latest", local_path="some/path"
        )

        mock_metoffice.return_value.process_files.assert_called_once_with()
        mock_metoffice.return_value.upload_to_geoserver.assert_called_once_with(
            styles=["jet"]
        )

        mock_remove_files.assert_called_once_with(
            local_path="some/path", data_type="atm", logger=main_function.logger
        )

    @patch("metoffice_data_handler.cli.main.remove_files")
    @patch("metoffice_data_handler.cli.main.MetOfficeInnovations")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_innovations(
        self,
        mock_parse_args: argparse.Namespace,
        mock_metoffice: main_function.MetOfficeInnovations,
        mock_remove_files: main_function.remove_files,
    ):
        """Test the cli function for the innovations data.

        Args:
            mock_parse_args (argparse.Namespace): The mock for the argparse.Namespace.
            mock_metoffice (main_function.MetOfficeInnovations): The mock for the
                MetOfficeInnovations class.
            mock_remove_files (main_function.remove_files): The mock for the
                remove_files function.
        """

        # Simulate command-line arguments
        mock_parse_args.return_value = argparse.Namespace(
            data_type="innovations",
            worker_threads=1,
            model_date="20230101",
            local_path="some/path",
            styles="jet,viridis",
            file_type="increments,model_errors",
            variables="TEM,SAL",
            upload=True,
            convert=True,
            remove_files=True,
        )

        main_function.main()

        mock_metoffice.assert_called_once_with(
            local_path="some/path",
            model_date="20230101",
            file_type=["increments", "model_errors"],
            logger=main_function.logger,
            worker_threads=1,
        )
        mock_metoffice.return_value.download_files.assert_called_once_with()

        mock_metoffice.return_value.process_files.assert_called_once_with(
            variables=["TEM", "SAL"]
        )
        mock_metoffice.return_value.upload_to_geoserver.assert_called_once_with(
            styles=["jet", "viridis"]
        )

        mock_remove_files.assert_called_once_with(
            local_path="some/path", data_type="innovations", logger=main_function.logger
        )

    @patch("metoffice_data_handler.cli.main.remove_files")
    @patch("metoffice_data_handler.cli.main.MetOfficeOcean")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_ocean(
        self,
        mock_parse_args: argparse.Namespace,
        mock_metoffice: main_function.MetOfficeOcean,
        mock_remove_files: main_function.remove_files,
    ):
        """Test the cli function for the ocean data.

        Args:
            mock_parse_args (argparse.Namespace): The mock for the argparse.Namespace.
            mock_metoffice (main_function.MetOfficeOcean): The mock for the MetOfficeOcean class.
            mock_remove_files (main_function.remove_files): The mock for the remove_files function.
        """
        mock_parse_args.return_value = argparse.Namespace(
            data_type="ocean",
            local_path="some/path",
            models="model1,model2",
            model_date="20230101",
            output_type="hi",
            forecast_days="1,2",
            variables="TEM,SAL",
            worker_threads=2,
            styles="jet,viridis",
            file_type="increments,model_errors",
            upload=True,
            convert=True,
            remove_files=True,
        )
        main_function.main()

        mock_metoffice.assert_called_once_with(
            local_path="some/path",
            models=["model1", "model2"],
            model_date="20230101",
            output_type="hi",
            forecast_days=["1", "2"],
            variables=["TEM", "SAL"],
            logger=main_function.logger,
            worker_threads=2,
        )
        mock_metoffice.return_value.download_files.assert_called_once_with()

        mock_metoffice.return_value.process_files.assert_called_once_with()
        mock_metoffice.return_value.upload_to_geoserver.assert_called_once_with(
            styles=["jet", "viridis"]
        )

        mock_remove_files.assert_called_once_with(
            local_path="some/path", data_type="ocean", logger=main_function.logger
        )


if __name__ == "__main__":
    unittest.main()
