"""Position layer module.

Process the images in the position layer.
"""

import logging
import numpy as np
import pandas as pd
from dask.distributed import Client
from geopy.distance import geodesic
from shapely.geometry import Polygon
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.investigation_layer.investigation_layer import InvestigationLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.position_params import POSITION_LAYER_METHODS
from paidiverpy.models.position_params import CalculateCornersParams
from paidiverpy.utils.exceptions import raise_value_error


class PositionLayer(Paidiverpy):
    """Position layer class.

    This class processes the images in the position layer.

    Args:
        parameters (dict): The parameters for the step.
        config_params (dict | ConfigParams, optional): The configuration parameters.
            It can contain the following keys / attributes:
            - input_path (str): The path to the input files.
            - output_path (str): The path to the output files.
            - metadata_path (str): The path to the metadata file.
            - metadata_type (str): The type of the metadata file.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of n_jobs.
        config_file_path (str): The path to the configuration file.
        config (Configuration): The configuration object.
        metadata (MetadataParser): The metadata object.
        images (ImagesLayer): The images object.
        paidiverpy (Paidiverpy): The paidiverpy object.
        step_name (str): The name of the step.
        client (Client): The Dask client.
        config_index (int): The index of the configuration.
        add_new_step (bool): Whether to add a new step.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        parameters: dict,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        client: Client | None = None,
        config_index: int | None = None,
        add_new_step: bool = True,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        super().__init__(
            config_params=config_params,
            config_file_path=config_file_path,
            metadata=metadata,
            config=config,
            images=images,
            paidiverpy=paidiverpy,
            client=client,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )

        self.step_name = step_name
        self.config_index = self.config.add_step(config_index, parameters, step_class=PositionLayer)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.raise_error = self._calculate_raise_error()
        self.step_order = len(self.images.steps)
        self.layer_methods = POSITION_LAYER_METHODS
        self.add_new_step = add_new_step

    def run(self) -> None:
        """Run the resample layer steps on the images based on the configuration.

        Run the resample layer steps on the images based on the configuration.

        Raises:
            ValueError: The mode is not defined in the configuration file.
        """
        mode = self.step_metadata.get("mode")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, self.layer_methods, mode, False)
        try:
            metadata = method(self.step_order, test=test, params=params)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in position layer: %s", e)
            if self.raise_error:
                raise_value_error("Position layer step failed.")
            self.logger.error("Position layer step will be skipped.")
            return self.get_metadata(flag="all")
        if not self.add_new_step:
            return metadata
        if not test:
            self.set_metadata(metadata)
            self.step_name = f"position_{mode}" if not self.step_name else self.step_name
            self.images.add_step(
                step=self.step_name,
                step_metadata=self.step_metadata,
                metadata=self.get_metadata(),
                update_metadata=True,
                track_changes=self.track_changes,
            )
        return None

    def calculate_corners(self, step_order: int | None = None, params: CalculateCornersParams = None, test: bool = False) -> pd.DataFrame:
        """Calculate the corners of the images.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (CalculateCornersParams, optional): The parameters for the position.
        Defaults to CalculateCornersParams().
        """
        params = CalculateCornersParams() if params is None else params
        metadata = self.get_metadata()
        metadata.loc[:, "image-camera-pitch-degrees"] = metadata["image-camera-pitch-degrees"].abs()
        metadata.loc[:, "image-camera-roll-degrees"] = metadata["image-camera-roll-degrees"].abs()

        theta = params.theta
        omega = params.omega
        camera_distance = params.camera_distance

        metadata["approx_vertdim_m"] = 2 * (metadata["image-altitude-meters"] + camera_distance) * np.tan(np.radians(theta / 2))
        metadata["approx_horizdim_m"] = 2 * (metadata["image-altitude-meters"] + camera_distance) * np.tan(np.radians(omega / 2))
        metadata["approx_area_m2"] = (
            4 * ((metadata["image-altitude-meters"] + camera_distance) ** 2) * np.tan(np.radians(theta / 2)) * np.tan(np.radians(omega / 2))
        )
        metadata["headingoffset_rad"] = np.arctan(metadata["approx_horizdim_m"] / metadata["approx_vertdim_m"])
        metadata["cornerdist_m"] = 0.5 * metadata["approx_horizdim_m"] / np.sin(metadata["headingoffset_rad"])
        metadata["longpos_deg"] = metadata["image-longitude"] + 360

        metadata = PositionLayer.calculate_limits(metadata)
        n = pd.DataFrame(
            {
                "long_deg": [
                    metadata["TLcornerlong"].iloc[0],
                    metadata["TRcornerlong"].iloc[0],
                    metadata["BRcornerlong"].iloc[0],
                    metadata["BLcornerlong"].iloc[0],
                ],
                "lat_deg": [
                    metadata["TLcornerlat"].iloc[0],
                    metadata["TRcornerlat"].iloc[0],
                    metadata["BRcornerlat"].iloc[0],
                    metadata["BLcornerlat"].iloc[0],
                ],
            },
        )

        chn = np.append(
            Polygon(n).convex_hull.exterior.coords,
            [Polygon(n).convex_hull.exterior.coords[0]],
            axis=0,
        )
        coordsn = pd.DataFrame(chn, columns=["long_deg", "lat_deg"])

        metadata["overlap"] = 0
        metadata["polygon_m"] = Polygon(coordsn.values)
        for i in metadata.index[1:]:
            m = pd.DataFrame(
                {
                    "long_deg": [
                        metadata["TLcornerlong"].loc[i],
                        metadata["TRcornerlong"].loc[i],
                        metadata["BRcornerlong"].loc[i],
                        metadata["BLcornerlong"].loc[i],
                    ],
                    "lat_deg": [
                        metadata["TLcornerlat"].loc[i],
                        metadata["TRcornerlat"].loc[i],
                        metadata["BRcornerlat"].loc[i],
                        metadata["BLcornerlat"].loc[i],
                    ],
                },
            )

            chm = np.append(
                Polygon(m).convex_hull.exterior.coords,
                [Polygon(m).convex_hull.exterior.coords[0]],
                axis=0,
            )
            coordsm = pd.DataFrame(chm, columns=["long_deg", "lat_deg"])
            polygon_m = Polygon(coordsm.values)
            metadata.loc[i, "polygon_m"] = polygon_m
        self.set_metadata(metadata, flag=True)
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="polygon").run()
            return None
        return metadata

    @staticmethod
    def calculate_limits(metadata: pd.DataFrame) -> pd.DataFrame:
        """Calculate the corners.

        Args:
            metadata (pd.DataFrame): The metadata.

        Returns:
            pd.DataFrame: The metadata with the corners.
        """
        corner_columns = [
            "TRcornerlong",
            "TRcornerlat",
            "TLcornerlong",
            "TLcornerlat",
            "BLcornerlong",
            "BLcornerlat",
            "BRcornerlong",
            "BRcornerlat",
        ]
        metadata[corner_columns] = 0.0
        for i, row in metadata.iterrows():
            lat, lon, heading_deg, headingoffset_rad, cornerdist_m = row[
                [
                    "image-latitude",
                    "longpos_deg",
                    "heading_deg",
                    "headingoffset_rad",
                    "cornerdist_m",
                ]
            ]

            metadata.loc[i, "TRcornerlong"], metadata.loc[i, "TRcornerlat"] = PositionLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                0,
            )
            metadata.loc[i, "TLcornerlong"], metadata.loc[i, "TLcornerlat"] = PositionLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                -2 * headingoffset_rad * 180 / np.pi,
            )
            metadata.loc[i, "BLcornerlong"], metadata.loc[i, "BLcornerlat"] = PositionLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                180,
            )
            metadata.loc[i, "BRcornerlong"], metadata.loc[i, "BRcornerlat"] = PositionLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                180 - 2 * headingoffset_rad * 180 / np.pi,
            )
        return metadata

    @staticmethod
    def calculate_corner(
        lat: float,
        lon: float,
        heading_deg: float,
        headingoffset_rad: float,
        cornerdist_m: float,
        angle_offset: float,
    ) -> tuple:
        """Calculate the corner coordinates.

        Args:
            lat (float): The latitude.
            lon (float): The longitude.
            heading_deg (float): The heading in degrees.
            headingoffset_rad (float): The heading offset in radians.
            cornerdist_m (float): The corner distance in meters.
            angle_offset (float): The angle offset.

        Returns:
            tuple: The corner coordinates.
        """
        angle = (headingoffset_rad * 180 / np.pi) + heading_deg + angle_offset
        destination = geodesic(meters=cornerdist_m).destination((lat, lon), angle)
        return destination.longitude, destination.latitude
