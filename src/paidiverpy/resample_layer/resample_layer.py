""" Open raw image file
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from paidiverpy import Paidiverpy
from paidiverpy.config.resample_params import (
    RESAMPLE_LAYER_METHODS,
    ResampleDatetimeParams,
    ResampleDepthParams,
    ResampleAltitudeParams,
    ResamplePitchRollParams,
    ResampleOverlappingParams,
    ResampleFixedParams,
    ResamplePercentParams,
    ResampleRegionParams,
    ResampleObscureParams,
)


class ResampleLayer(Paidiverpy):
    def __init__(
        self,
        config_file_path=None,
        input_path=None,
        output_path=None,
        metadata_path=None,
        metadata_type=None,
        metadata=None,
        config=None,
        logger=None,
        images=None,
        paidiverpy=None,
        step_name=None,
        parameters=None,
        config_index=None,
        raise_error=False,
        verbose=True,
        n_jobs: int =1,
    ):

        super().__init__(
            config_file_path=config_file_path,
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata_type=metadata_type,
            metadata=metadata,
            config=config,
            logger=logger,
            images=images,
            paidiverpy=paidiverpy,
            raise_error=raise_error,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.config_index = config_index
        self.step_name = step_name or "sampling"
        if parameters:
            if not parameters.get("step_name"):
                parameters["step_name"] = self.step_name
            self.config_index = self.config.add_step(config_index, parameters)
        self.step_order = len(self.images.steps)
        self.step_metadata = self._calculate_steps_metadata(
            self.config.steps[self.config_index]
        )

    def run(self):
        mode = self.step_metadata.get("mode")
        if not mode:
            raise ValueError("The mode is not defined in the configuration file.")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, RESAMPLE_LAYER_METHODS, mode)
        metadata = method(self.step_order, test=test, params=params)
        if self.step_order == 0:
            return metadata
        if not test:
            new_metadata = self.get_metadata(flag="all")
            new_metadata.loc[new_metadata.index.isin(metadata.index), "flag"] = (
                metadata.flag
            )
            self.set_metadata(new_metadata)
            self.step_name = f"trim_{mode}" if not self.step_name else self.step_name
            self.images.add_step(
                step=self.step_name,
                step_metadata=self.step_metadata,
                metadata=self.get_metadata(),
                update_metadata=True,
            )

    def _by_percent(
        self,
        step_order=None,
        test=False,
        params: ResamplePercentParams = ResamplePercentParams(),
    ):
        metadata = self.get_metadata().sample(frac=1, random_state=np.random.seed())
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
        step_order=None,
        test=False,
        params: ResampleFixedParams = ResampleFixedParams(),
    ):
        metadata = self.get_metadata().sample(frac=1, random_state=np.random.seed())
        if params.value > len(self.get_metadata()):
            self.logger.info(
                "Number of photos to be removed is greater than the number of photos in the metadata."
            )
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
        step_order=None,
        test=False,
        params: ResampleDatetimeParams = ResampleDatetimeParams(),
    ):
        metadata = self.get_metadata()
        start_date = params.min
        end_date = params.max
        if not start_date and not end_date:
            return metadata
        if start_date is None:
            start_date = metadata["datetime"].min()
        else:
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = metadata["datetime"].max()
        else:
            end_date = pd.to_datetime(end_date)
        if start_date > end_date:
            raise ValueError("Start date cannot be greater than end date")
        if step_order == 0:
            return metadata.loc[
                (metadata["datetime"] >= start_date) & (metadata["datetime"] <= end_date)
            ]
        metadata.loc[
            (metadata["datetime"] < start_date) | (metadata["datetime"] > end_date),
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
        step_order=None,
        test=False,
        params: ResampleDepthParams = ResampleDepthParams(),
    ):
        metadata = self.get_metadata()
        metadata.loc[:, "depth_m"] = metadata["depth_m"].abs()
        if params.by == "lower":
            metadata.loc[metadata["depth_m"] < params.value, "flag"] = step_order
        else:
            metadata.loc[metadata["depth_m"] > params.value, "flag"] = step_order
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
        step_order=None,
        test=False,
        params: ResampleAltitudeParams = ResampleAltitudeParams(),
    ):
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
        step_order=None,
        test=False,
        params: ResamplePitchRollParams = ResamplePitchRollParams(),
    ):
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
        step_order=None,
        test=False,
        params: ResampleRegionParams = ResampleRegionParams(),
    ):
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
                        ]
                    )
                ]
            )

        def point_in_any_polygon(point):
            return any(polygon.contains(point) for polygon in polygons.geometry)

        metadata["flag"] = metadata.apply(
            lambda x: x["flag"] if point_in_any_polygon(x["point"]) else step_order,
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
        step_order=None,
        test=False,
        params: ResampleObscureParams = ResampleObscureParams(),
    ):
        metadata = self.get_metadata()
        images = self.images.get_step(step=self.config_index, by_order=True)

        def compute_mean(image_chunk):
            return np.mean(image_chunk, axis=(1, 2)) / 255
        if self.n_jobs == 1:
            brightness = compute_mean(images)
        else:
            brightness = images.map_blocks(compute_mean, dtype=float)
        
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
            plt.show()
            return None
        return metadata

    def _by_overlapping(
        self,
        step_order=None,
        test=False,
        params: ResampleOverlappingParams = ResampleOverlappingParams(),
    ):
        metadata = self.get_metadata()
        metadata.loc[:, "pitch_deg"] = metadata["pitch_deg"].abs()
        metadata.loc[:, "roll_deg"] = metadata["roll_deg"].abs()

        theta = params.theta
        omega = params.omega
        overlap_threshold = params.threshold

        # TODO change de 1.12 to a parameter (distance between camera and the altimeter)
        metadata["approx_vertdim_m"] = (
            2 * (metadata["altitude_m"] + 1.12) * np.tan(np.radians(theta / 2))
        )
        metadata["approx_horizdim_m"] = (
            2 * (metadata["altitude_m"] + 1.12) * np.tan(np.radians(omega / 2))
        )
        metadata["approx_area_m2"] = (
            4
            * ((metadata["altitude_m"] + 1.12) ** 2)
            * np.tan(np.radians(theta / 2))
            * np.tan(np.radians(omega / 2))
        )
        metadata["headingoffset_rad"] = np.arctan(
            metadata["approx_horizdim_m"] / metadata["approx_vertdim_m"]
        )
        metadata["cornerdist_m"] = (
            0.5 * metadata["approx_horizdim_m"] / np.sin(metadata["headingoffset_rad"])
        )
        metadata["longpos_deg"] = metadata["lon"] + 360

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
        # Iterate over each photo to calculate corner coordinates
        for i, row in metadata.iterrows():
            lat, lon, heading_deg, headingoffset_rad, cornerdist_m = row[
                [
                    "lat",
                    "longpos_deg",
                    "heading_deg",
                    "headingoffset_rad",
                    "cornerdist_m",
                ]
            ]

            metadata.loc[i, "TRcornerlong"], metadata.loc[i, "TRcornerlat"] = (
                ResampleLayer.calculate_corner(
                    lat, lon, heading_deg, headingoffset_rad, cornerdist_m, 0
                )
            )
            metadata.loc[i, "TLcornerlong"], metadata.loc[i, "TLcornerlat"] = (
                ResampleLayer.calculate_corner(
                    lat,
                    lon,
                    heading_deg,
                    headingoffset_rad,
                    cornerdist_m,
                    -2 * headingoffset_rad * 180 / np.pi,
                )
            )
            metadata.loc[i, "BLcornerlong"], metadata.loc[i, "BLcornerlat"] = (
                ResampleLayer.calculate_corner(
                    lat, lon, heading_deg, headingoffset_rad, cornerdist_m, 180
                )
            )
            metadata.loc[i, "BRcornerlong"], metadata.loc[i, "BRcornerlat"] = (
                ResampleLayer.calculate_corner(
                    lat,
                    lon,
                    heading_deg,
                    headingoffset_rad,
                    cornerdist_m,
                    180 - 2 * headingoffset_rad * 180 / np.pi,
                )
            )

        # Calculate the coordinates for the first photo
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
            }
        )

        chn = np.append(
            Polygon(n).convex_hull.exterior.coords,
            [Polygon(n).convex_hull.exterior.coords[0]],
            axis=0,
        )
        coordsn = pd.DataFrame(chn, columns=["long_deg", "lat_deg"])

        # Find overlaps
        metadata["overlap"] = 0
        metadata["polygon_m"] = Polygon(coordsn.values)

        for i in range(1, len(metadata)):
            m = pd.DataFrame(
                {
                    "long_deg": [
                        metadata["TLcornerlong"].iloc[i],
                        metadata["TRcornerlong"].iloc[i],
                        metadata["BRcornerlong"].iloc[i],
                        metadata["BLcornerlong"].iloc[i],
                    ],
                    "lat_deg": [
                        metadata["TLcornerlat"].iloc[i],
                        metadata["TRcornerlat"].iloc[i],
                        metadata["BRcornerlat"].iloc[i],
                        metadata["BLcornerlat"].iloc[i],
                    ],
                }
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
                    if (
                        overlap_percentage_n > overlap_threshold
                        or overlap_percentage_m > overlap_threshold
                    ):
                        metadata.loc[i, "overlap"] = 1
                    else:
                        coordsn = coordsm
            else:
                coordsn = coordsm
        self.logger.info(
            "Number of photos to be removed: %s", int(metadata["overlap"].sum())
        )
        new_metadata = self.get_metadata()

        new_metadata.loc[metadata["overlap"] == 1, "flag"] = step_order
        if test:
            ResampleLayer.plot_polygons(metadata)
            self.plot_trimmed_photos(new_metadata[new_metadata.flag == 0])
            return None
        return new_metadata

    @staticmethod
    def plot_polygons(metadata):
        gdf = gpd.GeoDataFrame(metadata, geometry="polygon_m")
        _, ax = plt.subplots(figsize=(15, 15))

        gdf[gdf.overlap == 0].plot(
            ax=ax, facecolor="none", edgecolor="black", label="No Overlap"
        )
        gdf[gdf.overlap == 1].plot(
            ax=ax, facecolor="none", edgecolor="red", label="Overlap"
        )
        plt.savefig("overlap.png")
        plt.show()

    @staticmethod
    def calculate_corner(
        lat, lon, heading_deg, headingoffset_rad, cornerdist_m, angle_offset
    ):
        angle = (headingoffset_rad * 180 / np.pi) + heading_deg + angle_offset
        destination = geodesic(meters=cornerdist_m).destination((lat, lon), angle)
        return destination.longitude, destination.latitude
