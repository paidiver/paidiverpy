"""ResampleLayer class.

Resample the images based on the configuration file.
"""

import logging
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client
from geopy.distance import geodesic
from shapely.geometry import Polygon
from paidiverpy import Paidiverpy
from paidiverpy.config.config import Configuration
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.resample_params import RESAMPLE_LAYER_METHODS
from paidiverpy.config.resample_params import ResampleAltitudeParams
from paidiverpy.config.resample_params import ResampleDatetimeParams
from paidiverpy.config.resample_params import ResampleDepthParams
from paidiverpy.config.resample_params import ResampleFixedParams
from paidiverpy.config.resample_params import ResampleObscureParams
from paidiverpy.config.resample_params import ResampleOverlappingParams
from paidiverpy.config.resample_params import ResamplePercentParams
from paidiverpy.config.resample_params import ResamplePitchRollParams
from paidiverpy.config.resample_params import ResampleRegionParams
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils.exceptions import raise_value_error


class ResampleLayer(Paidiverpy):
    """Process the images in the resample layer.

    Args:
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
        parameters (dict): The parameters for the step.
        client (Client): The Dask client.
        config_index (int): The index of the configuration.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_name: str | None = None,
        parameters: dict | None = None,
        client: Client | None = None,
        config_index: int | None = None,
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
        self.step_name = step_name or "sampling"
        if parameters:
            if not parameters.get("step_name"):
                parameters["step_name"] = self.step_name
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_order = len(self.images.steps)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])

    def run(self) -> None:
        """Run the resample layer steps on the images based on the configuration.

        Run the resample layer steps on the images based on the configuration.

        Raises:
            ValueError: The mode is not defined in the configuration file.
        """
        mode = self.step_metadata.get("mode")
        if not mode:
            msg = "The mode is not defined in the configuration file."
            raise ValueError(msg)
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, RESAMPLE_LAYER_METHODS, mode, False)
        try:
            metadata = method(self.step_order, test=test, params=params)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in resample layer: %s", e)
            if self.raise_error:
                raise_value_error("Resample layer step failed.")
            self.logger.error("Resample layer step will be skipped.")
            metadata = self.get_metadata(flag="all")
        if self.step_order == 0:
            return metadata
        if not test:
            new_metadata = self.get_metadata(flag="all")
            new_metadata.loc[new_metadata.index.isin(metadata.index), "flag"] = metadata.flag
            self.set_metadata(new_metadata)
            self.step_name = f"trim_{mode}" if not self.step_name else self.step_name
            self.images.add_step(
                step=self.step_name,
                step_metadata=self.step_metadata,
                metadata=self.get_metadata(),
                update_metadata=True,
                track_changes=self.track_changes,
            )
            return None
        return None

    def _by_percent(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResamplePercentParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by a percentage.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResamplePercentParams, optional): The parameters for the resample.
        Defaults to ResamplePercentParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResamplePercentParams()
        metadata = self.get_metadata().sample(frac=1, random_state=np.random.default_rng())
        new_metadata = metadata.sample(frac=params.value)
        if step_order == 0:
            return new_metadata
        if test:
            self.plot_trimmed_photos(new_metadata)
            return None
        metadata.loc[~metadata.index.isin(new_metadata.index), "flag"] = step_order
        return metadata

    def _by_fixed_number(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleFixedParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by a fixed number of photos.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleFixedParams, optional): The parameters for the resample.
        Defaults to ResampleFixedParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResampleFixedParams()
        metadata = self.get_metadata().sample(frac=1, random_state=np.random.default_rng())
        if params.value > len(self.get_metadata()):
            self.logger.info("Number of photos to be removed is greater than the number of photos in the metadata.")
            self.logger.info("No photos will be removed.")
            return self.get_metadata()
        new_metadata = metadata.sample(n=params.value)
        if step_order == 0:
            return new_metadata
        if test:
            self.plot_trimmed_photos(new_metadata)
            return None
        metadata.loc[~metadata.index.isin(new_metadata.index), "flag"] = step_order
        return metadata

    def _by_datetime(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleDatetimeParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by datetime.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleDatetimeParams, optional): The parameters for the resample.
        Defaults to ResampleDatetimeParams().

        Raises:
            ValueError: Start date cannot be greater than end date.

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResampleDatetimeParams()
        metadata = self.get_metadata()
        start_date = params.min
        end_date = params.max
        if not start_date and not end_date:
            return metadata
        start_date = metadata["image-datetime"].min() if start_date is None else pd.to_datetime(start_date)
        end_date = metadata["image-datetime"].max() if end_date is None else pd.to_datetime(end_date)
        if start_date > end_date:
            msg = "Start date cannot be greater than end date"
            raise ValueError(msg)
        if step_order == 0:
            return metadata.loc[(metadata["image-datetime"] >= start_date) & (metadata["image-datetime"] <= end_date)]
        metadata.loc[
            (metadata["image-datetime"] < start_date) | (metadata["image-datetime"] > end_date),
            "flag",
        ] = step_order
        self.logger.info(
            "Number of photos to be removed: %s",
            metadata.flag[metadata.flag == step_order].count(),
        )
        if test:
            self.plot_trimmed_photos(metadata[metadata.flag == 0])
            return None
        return metadata

    def _by_depth(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleDepthParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by depth.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleDepthParams, optional): The parameters for the resample.
        Defaults to ResampleDepthParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResampleDepthParams()
        metadata = self.get_metadata()
        metadata.loc[:, "image-altitude-meters"] = metadata["image-altitude-meters"].abs()
        if params.by == "lower":
            metadata.loc[metadata["image-altitude-meters"] < params.value, "flag"] = step_order
        else:
            metadata.loc[metadata["image-altitude-meters"] > params.value, "flag"] = step_order
        self.logger.info(
            "Number of photos to be removed: %s",
            metadata.flag[metadata.flag == step_order].count(),
        )
        if test:
            self.plot_trimmed_photos(metadata[metadata.flag == 0])
            return None
        return metadata

    def _by_altitude(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleAltitudeParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by altitude.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleAltitudeParams, optional): The parameters for the resample.
        Defaults to ResampleAltitudeParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResampleAltitudeParams()
        metadata = self.get_metadata()
        metadata.loc[:, "altitude_m"] = metadata["altitude_m"].abs()
        metadata.loc[metadata["altitude_m"] > params.value, "flag"] = step_order
        self.logger.info(
            "Number of photos to be removed: %s",
            metadata.flag[metadata.flag == step_order].count(),
        )
        if test:
            self.plot_trimmed_photos(metadata[metadata.flag == 0])
            return None
        return metadata

    def _by_pitch_roll(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResamplePitchRollParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by pitch and roll.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResamplePitchRollParams, optional): The parameters for the resample.
        Defaults to ResamplePitchRollParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResamplePitchRollParams()
        metadata = self.get_metadata()
        metadata.loc[:, "pitch_deg"] = metadata["pitch_deg"].abs()
        metadata.loc[:, "roll_deg"] = metadata["roll_deg"].abs()
        metadata.loc[
            (metadata["pitch_deg"] > params.pitch) & (metadata["roll_deg"] > params.roll),
            "flag",
        ] = step_order
        self.logger.info(
            "Number of photos to be removed: %s",
            metadata.flag[metadata.flag == step_order].count(),
        )
        if test:
            self.plot_trimmed_photos(metadata[metadata.flag == 0])
            return None
        return metadata

    def _by_region(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleRegionParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by region.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleRegionParams, optional): The parameters for the resample.
        Defaults to ResampleRegionParams().

        Returns:
            pd.DataFrame: _description_
        """
        if params is None:
            params = ResampleRegionParams()
        metadata = self.get_metadata()
        if params.file:
            polygons = gpd.read_file(params.file)
        else:
            min_lon, max_lon, min_lat, max_lat = params.limits
            polygons = gpd.GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            (min_lon, min_lat),
                            (min_lon, max_lat),
                            (max_lon, max_lat),
                            (max_lon, min_lat),
                        ],
                    ),
                ],
            )

        def _point_in_any_polygon(point: tuple) -> bool:
            return any(polygon.contains(point) for polygon in polygons.geometry)

        metadata["flag"] = metadata.apply(
            lambda x: x["flag"] if _point_in_any_polygon(x["point"]) else step_order,
            axis=1,
        )
        self.logger.info(
            "Number of photos to be removed: %s",
            metadata.flag[metadata.flag == step_order].count(),
        )
        if test:
            self.plot_trimmed_photos(metadata[metadata.flag == 0])
            return None
        return metadata

    def _by_obscure_photos(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleObscureParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by obscure photos.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleObscureParams, optional): The parameters for the resample.
        Defaults to ResampleObscureParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResampleObscureParams()
        metadata = self.get_metadata()
        images = self.images.get_step(step=self.config_index, by_order=True)

        def _compute_mean(image_chunk: np.ndarray) -> np.ndarray:
            return np.mean(image_chunk, axis=(1, 2)) / 255

        brightness = _compute_mean(images) if self.n_jobs == 1 else images.map_blocks(_compute_mean, dtype=float)

        metadata.loc[brightness < params.min, "flag"] = step_order
        metadata.loc[brightness > params.max, "flag"] = step_order
        self.logger.info(
            "Number of photos to be removed: %s",
            metadata.flag[metadata.flag == step_order].count(),
        )
        if test:
            self.plot_trimmed_photos(metadata[metadata.flag == 0])
            plt.hist(brightness, bins=30, edgecolor="black")
            plt.xlabel("Mean RGB Brightness")
            plt.ylabel("Frequency")
            plt.title("Distribution of Image Brightness")
            plt.show(block=False)
            return None
        return metadata

    def _by_overlapping(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: ResampleOverlappingParams = None,
    ) -> pd.DataFrame:
        """Resample the metadata by overlapping photos.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (ResampleOverlappingParams, optional): The parameters for the resample.
        Defaults to ResampleOverlappingParams().

        Returns:
            pd.DataFrame: Metadata with the photos to be removed flagged.
        """
        if params is None:
            params = ResampleOverlappingParams()
        metadata = self.get_metadata()

        metadata.loc[:, "pitch_deg"] = metadata["pitch_deg"].abs()
        metadata.loc[:, "roll_deg"] = metadata["roll_deg"].abs()

        theta = params.theta
        omega = params.omega
        overlap_threshold = params.threshold
        camera_distance = params.camera_distance

        metadata["approx_vertdim_m"] = 2 * (metadata["altitude_m"] + camera_distance) * np.tan(np.radians(theta / 2))
        metadata["approx_horizdim_m"] = 2 * (metadata["altitude_m"] + camera_distance) * np.tan(np.radians(omega / 2))
        metadata["approx_area_m2"] = (
            4 * ((metadata["altitude_m"] + camera_distance) ** 2) * np.tan(np.radians(theta / 2)) * np.tan(np.radians(omega / 2))
        )
        metadata["headingoffset_rad"] = np.arctan(metadata["approx_horizdim_m"] / metadata["approx_vertdim_m"])
        metadata["cornerdist_m"] = 0.5 * metadata["approx_horizdim_m"] / np.sin(metadata["headingoffset_rad"])
        metadata["longpos_deg"] = metadata["image-longitude"] + 360

        metadata = ResampleLayer.calculate_corners(metadata)
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

            polygon_n = Polygon(coordsn.values)
            polygon_m = Polygon(coordsm.values)
            metadata.loc[i, "polygon_m"] = polygon_m
            if polygon_n.intersects(polygon_m):
                if overlap_threshold is None:
                    metadata.loc[i, "overlap"] = 1
                else:
                    overlap_area = polygon_n.intersection(polygon_m).area
                    overlap_percentage_n = overlap_area / polygon_n.area
                    overlap_percentage_m = overlap_area / polygon_m.area
                    if overlap_percentage_n > overlap_threshold or overlap_percentage_m > overlap_threshold:
                        metadata.loc[i, "overlap"] = 1
                    else:
                        coordsn = coordsm
                        metadata.loc[i, "overlap"] = 0
            else:
                coordsn = coordsm
                metadata.loc[i, "overlap"] = 0
        self.logger.info("Number of photos to be removed: %s", int(metadata["overlap"].sum()))
        new_metadata = self.get_metadata()
        new_metadata.loc[metadata["overlap"] == 1, "flag"] = step_order
        if test:
            ResampleLayer.plot_polygons(metadata)
            self.plot_trimmed_photos(new_metadata[new_metadata.flag == 0])
            return None
        return new_metadata

    @staticmethod
    def plot_polygons(metadata: pd.DataFrame) -> None:
        """Plot the polygons.

        Args:
            metadata (pd.DataFrame): The metadata with the polygons.
        """
        gdf = gpd.GeoDataFrame(metadata, geometry="polygon_m")
        _, ax = plt.subplots(figsize=(15, 15))

        gdf[gdf.overlap == 0].plot(ax=ax, facecolor="none", edgecolor="black", label="No Overlap")
        gdf[gdf.overlap == 1].plot(ax=ax, facecolor="none", edgecolor="red", label="Overlap")
        plt.show(block=False)

    @staticmethod
    def calculate_corners(metadata: pd.DataFrame) -> pd.DataFrame:
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

            metadata.loc[i, "TRcornerlong"], metadata.loc[i, "TRcornerlat"] = ResampleLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                0,
            )
            metadata.loc[i, "TLcornerlong"], metadata.loc[i, "TLcornerlat"] = ResampleLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                -2 * headingoffset_rad * 180 / np.pi,
            )
            metadata.loc[i, "BLcornerlong"], metadata.loc[i, "BLcornerlat"] = ResampleLayer.calculate_corner(
                lat,
                lon,
                heading_deg,
                headingoffset_rad,
                cornerdist_m,
                180,
            )
            metadata.loc[i, "BRcornerlong"], metadata.loc[i, "BRcornerlat"] = ResampleLayer.calculate_corner(
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
