"""SamplingLayer class.

Sampling the images based on the configuration file.
"""

import logging
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from shapely.geometry import Polygon
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.investigation_layer.investigation_layer import InvestigationLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.models.sampling_params import SAMPLING_LAYER_METHODS
from paidiverpy.models.sampling_params import SamplingAltitudeParams
from paidiverpy.models.sampling_params import SamplingDatetimeParams
from paidiverpy.models.sampling_params import SamplingDepthParams
from paidiverpy.models.sampling_params import SamplingFixedParams
from paidiverpy.models.sampling_params import SamplingObscureParams
from paidiverpy.models.sampling_params import SamplingOverlappingParams
from paidiverpy.models.sampling_params import SamplingPercentParams
from paidiverpy.models.sampling_params import SamplingPitchRollParams
from paidiverpy.models.sampling_params import SamplingRegionParams
from paidiverpy.position_layer.position_layer import PositionLayer
from paidiverpy.utils.data import EIGHT_BITS_SIZE
from paidiverpy.utils.data import NUM_CHANNELS_RGB
from paidiverpy.utils.data import SIXTEEN_BITS_SIZE
from paidiverpy.utils.data import THIRTY_TWO_BITS_SIZE
from paidiverpy.utils.exceptions import raise_value_error


class SamplingLayer(Paidiverpy):
    """Process the images in the resample layer.

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
        if not parameters.get("mode"):
            msg = "Mode is not defined for the resample layer."
            self.logger.error(msg)
            raise_value_error(msg)
        self.step_name = step_name or "sampling"
        if not parameters.get("step_name"):
            parameters["step_name"] = self.step_name
        self.config_index = self.config.add_step(config_index=config_index, parameters=parameters, step_class=SamplingLayer)
        self.step_order = len(self.images.steps)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])
        self.add_new_step = add_new_step
        self.raise_error = self._calculate_raise_error()

    def run(self) -> None:
        """Run the resample layer steps on the images based on the configuration.

        Run the resample layer steps on the images based on the configuration.

        Raises:
            ValueError: The mode is not defined in the configuration file.
        """
        mode = self.step_metadata.get("mode")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, SAMPLING_LAYER_METHODS, mode, False)
        try:
            step_order = 9999 if self.step_order == 0 else self.step_order
            metadata = method(step_order, test=test, params=params)
            new_metadata = self.get_metadata(flag="all")
            if not self.add_new_step:
                metadata = metadata.loc[metadata["flag"] == 0]
                metadata = MetadataParser.df2dataarray(metadata)
                metadata.attrs["dataset_metadata"] = new_metadata.attrs["dataset_metadata"]
                return metadata
            metadata = MetadataParser.df2dataarray(metadata)
            metadata_flag = metadata["flag"].to_numpy()
            new_metadata = xr.where(new_metadata["flag"] == 0, metadata, new_metadata)
            new_metadata = new_metadata.assign_coords(flag=("filename", metadata_flag))
            number_of_images_in_this_step = len(metadata_flag)
            keep_images = len(metadata_flag[metadata_flag == 0])
            self.logger.info(
                "Number of images to be removed: %s. Total number of images: %s",
                number_of_images_in_this_step - keep_images,
                number_of_images_in_this_step,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in resample layer: %s", e)
            if self.raise_error:
                raise_value_error("Sampling layer step failed.")
            self.logger.error("Sampling layer step will be skipped.")
        if not test and self.add_new_step:
            self.step_name = f"trim_{mode}" if not self.step_name else self.step_name
            image_ds = self.images.get_step(last=True)
            self.images.add_step(
                step=self.step_name,
                images=image_ds,
                step_metadata=self.step_metadata,
                metadata=new_metadata,
                track_changes=self.track_changes,
            )
        return None

    def _by_percent(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingPercentParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by a percentage.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingPercentParams, optional): The parameters for the resample.
        Defaults to SamplingPercentParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingPercentParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        flagged_index = metadata.sample(frac=(1 - params.value)).index
        metadata.loc[flagged_index, "flag"] = step_order
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()
        return metadata

    def _by_fixed_number(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingFixedParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by a fixed number of images.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingFixedParams, optional): The parameters for the resample.
        Defaults to SamplingFixedParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingFixedParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        if params.value >= len(metadata):
            self.logger.info("Number of images to be removed is greater than the number of images in the metadata.")
            self.logger.info("No images will be removed.")
        else:
            flagged_index = metadata.sample(n=(len(metadata) - params.value)).index
            metadata.loc[flagged_index, "flag"] = step_order
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()
        return metadata

    def _by_datetime(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingDatetimeParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by datetime.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingDatetimeParams, optional): The parameters for the resample.
        Defaults to SamplingDatetimeParams().

        Raises:
            ValueError: Start date cannot be greater than end date.

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingDatetimeParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")

        start_date = params.min
        end_date = params.max
        if not start_date and not end_date:
            self.logger.info("No start or end date provided. No images will be removed.")
        else:
            start_date = metadata["image-datetime"].min() if start_date is None else pd.to_datetime(start_date)
            end_date = metadata["image-datetime"].max() if end_date is None else pd.to_datetime(end_date)
            if start_date > end_date:
                msg = "Start date cannot be greater than end date"
                raise ValueError(msg)
            metadata.loc[~((metadata["image-datetime"] >= start_date) & (metadata["image-datetime"] <= end_date)), "flag"] = step_order
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()

        return metadata

    def _by_depth(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingDepthParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by depth.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingDepthParams, optional): The parameters for the resample.
        Defaults to SamplingDepthParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingDepthParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        metadata.loc[:, "image-depth"] = metadata["image-depth"].abs()
        if params.by == "lower":
            metadata.loc[metadata["image-depth"] <= params.value]["flag"] = step_order
        else:
            metadata.loc[metadata["image-depth"] >= params.value]["flag"] = step_order
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()
        return metadata

    def _by_altitude(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingAltitudeParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by altitude.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingAltitudeParams, optional): The parameters for the resample.
        Defaults to SamplingAltitudeParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingAltitudeParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        metadata.loc[:, "image-altitude-meters"] = metadata["image-altitude-meters"].abs()
        if params.by == "lower":
            metadata.loc[metadata["image-altitude-meters"] <= params.value]["flag"] = step_order
        else:
            metadata.loc[metadata["image-altitude-meters"] >= params.value]["flag"] = step_order
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()
        return metadata

    def _by_pitch_roll(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingPitchRollParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by pitch and roll.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingPitchRollParams, optional): The parameters for the resample.
        Defaults to SamplingPitchRollParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingPitchRollParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        metadata.loc[:, "image-camera-pitch-degrees"] = metadata["image-camera-pitch-degrees"].abs()
        metadata.loc[:, "image-camera-roll-degrees"] = metadata["image-camera-roll-degrees"].abs()

        metadata.loc[((metadata["image-camera-pitch-degrees"] > params.pitch) & (metadata["image-camera-roll-degrees"] > params.roll))]["flag"] = (
            step_order
        )
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()
        return metadata

    def _by_region(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingRegionParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by region.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingRegionParams, optional): The parameters for the resample.
        Defaults to SamplingRegionParams().

        Returns:
            pd.DataFrame: _description_
        """
        params = SamplingRegionParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        if not params.limits and not params.file:
            self.logger.info("No limits or file provided. No images will be removed.")
        else:
            if params.file:
                polygons = gpd.read_file(params.file)
            else:
                polygons = gpd.GeoDataFrame(
                    geometry=[
                        Polygon(
                            [
                                (params.limits["min_lon"], params.limits["min_lat"]),
                                (params.limits["min_lon"], params.limits["max_lat"]),
                                (params.limits["max_lon"], params.limits["max_lat"]),
                                (params.limits["max_lon"], params.limits["min_lat"]),
                            ],
                        ),
                    ],
                )

            def _point_in_any_polygon(point: tuple) -> bool:
                return any(polygon.contains(point) for polygon in polygons.geometry)

            metadata.loc[~metadata["point"].apply(_point_in_any_polygon), "flag"] = step_order
            if self.images.images is None:
                dataset_metadata = self.metadata.metadata.attrs["dataset_metadata"]
                dataset_metadata["trimmed_polygon"] = polygons.geometry
                self.metadata.metadata.attrs["dataset_metadata"] = dataset_metadata
            else:
                dataset_metadata = self.images.images.attrs["dataset_metadata"]
                dataset_metadata["trimmed_polygon"] = polygons.geometry
                self.images.images.attrs["dataset_metadata"] = dataset_metadata
        if test:
            InvestigationLayer(paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample").run()
        return metadata

    def _by_obscure_images(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingObscureParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by obscure images.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingObscureParams, optional): The parameters for the resample.
        Defaults to SamplingObscureParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingObscureParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        if params.max < params.min:
            self.logger.error("Max value cannot be less than min value.")
            self.logger.error("No images will be removed.")
            return metadata
        images = self.images.get_step(step=self.config_index)
        bits = images[0].dtype.itemsize
        if self.n_jobs == 1:
            brightness = SamplingLayer._compute_mean(images, bits)
        else:
            images.map_blocks(SamplingLayer._compute_mean, bits=bits, meta=np.float32)
        if brightness.ndim == 1:
            brightness = np.expand_dims(brightness, axis=1)
        brightness = brightness[:, :3] if brightness.shape[1] > NUM_CHANNELS_RGB else brightness
        if brightness.shape[1] == 1:
            brightness = brightness[:, 0]
            metadata["brightness"] = brightness
            mask = ~((metadata["brightness"] > params.min) & (metadata["brightness"] < params.max))
            metadata.loc[mask, "flag"] = step_order
        elif params.channel == "mean":
            metadata["brightness"] = np.mean(brightness, axis=1)
            mask = ~((metadata["brightness"] > params.min) & (metadata["brightness"] < params.max))
            metadata.loc[mask, "flag"] = step_order
        else:
            for channel in range(brightness.shape[1]):
                metadata[f"brightness_{channel + 1}"] = brightness[:, channel]
            if params.channel == "all":
                for channel in range(brightness.shape[1]):
                    mask = ~((metadata[f"brightness_{channel + 1}"] > params.min) & (metadata[f"brightness_{channel + 1}"] < params.max))
                    metadata.loc[mask, "flag"] = step_order
            else:
                mask = ~((metadata[f"brightness_{params.channel}"] > params.min) & (metadata[f"brightness_{params.channel}"] < params.max))
                metadata.loc[mask, "flag"] = step_order
        if test:
            InvestigationLayer(
                paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample-obscure"
            ).run()
        return metadata

    def _by_overlapping(
        self,
        step_order: int | None = None,
        test: bool = False,
        params: SamplingOverlappingParams = None,
    ) -> pd.DataFrame:
        """Sampling the metadata by overlapping images.

        Args:
            step_order (int, optional): The order of the step. Defaults to None.
            test (bool, optional): Whether to test the step. Defaults to False.
            params (SamplingOverlappingParams, optional): The parameters for the resample.
        Defaults to SamplingOverlappingParams().

        Returns:
            pd.DataFrame: Metadata with the images to be removed flagged.
        """
        params = SamplingOverlappingParams() if params is None else params
        metadata = self.get_metadata(output_format="pandas")
        theta = params.theta
        omega = params.omega
        overlap_threshold = params.threshold
        camera_distance = params.camera_distance

        if "polygon_m" not in metadata.columns:
            # new_config = copy.copy(self.config)
            step_params = {
                "step_name": "position",
                "name": "corners",
                "mode": "calculate_corners",
                "params": {
                    "theta": theta,
                    "omega": omega,
                    "camera_distance": camera_distance,
                },
            }
            metadata = PositionLayer(
                paidiverpy=self,
                # config=new_config,
                # metadata=self.metadata,
                parameters=step_params,
                # client=self.client,
                add_new_step=False,
            ).run()
        else:
            metadata = self.get_metadata(output_format="pandas")
        metadata["overlap"] = 0
        index_comparison = 0
        for i in metadata.index[1:]:
            polygon_n = metadata.loc[index_comparison, "polygon_m"]
            polygon_m = metadata.loc[i, "polygon_m"]
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
                        index_comparison = i
                        metadata.loc[i, "overlap"] = 0
            else:
                index_comparison = i
                metadata.loc[i, "overlap"] = 0
        metadata.loc[metadata["overlap"] == 1]["flag"] = step_order
        if test:
            InvestigationLayer(
                paidiverpy=self, step_order=step_order, step_name=self.step_name, plot_metadata=metadata, plots="resample-polygon"
            ).run()
        return metadata

    @staticmethod
    def _compute_mean(image_chunk: np.ndarray, bits: int) -> np.ndarray:
        if bits == EIGHT_BITS_SIZE:
            result = np.mean(image_chunk, axis=(1, 2)) / 255
        elif bits == SIXTEEN_BITS_SIZE:
            result = np.mean(image_chunk, axis=(1, 2)) / 65535
        elif bits == THIRTY_TWO_BITS_SIZE:
            result = np.mean(image_chunk, axis=(1, 2)) / 4294967295
        else:
            result = np.mean(image_chunk, axis=(1, 2)) / 255
        return result
