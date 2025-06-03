"""Position layer module.

Process the images in the position layer.
"""

import logging
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser


class InvestigationLayer(Paidiverpy):
    """Investigation layer class.

    This class processes the images in the position layer.

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
        config_index (int): The index of the configuration.
        logger (logging.Logger): The logger object.
        raise_error (bool): Whether to raise an error.
        verbose (int): verbose level (0 = none, 1 = errors/warnings, 2 = info).
        plots (list | str): The plots to generate.
        plot_metadata (pd.DataFrame): The metadata for the
    """

    def __init__(
        self,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        paidiverpy: "Paidiverpy" = None,
        step_order: str | None = None,
        step_name: str | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
        plots: list | str | None = None,
        plot_metadata: pd.DataFrame = None,
    ):
        super().__init__(
            config_params=config_params,
            config_file_path=config_file_path,
            metadata=metadata,
            config=config,
            images=images,
            paidiverpy=paidiverpy,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )
        self.step_order = step_order
        self.step_name = step_name
        self.plots = plots
        self.plot_metadata = plot_metadata
        self.output_path, self.is_remote = self.config.get_output_path()

    def run(self) -> None:
        """Run the investigation layer."""
        self.logger.warning("As you are using the test mode, the investigation layer will be executed.")
        if self.is_remote:
            self.logger.error("Output path is remote. Skipping investigation layer.")
            return
        self.output_path = self.output_path / f"{self.step_order}_{self.step_name}"
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.plot_metadata is None:
            self.plot_metadata = self.get_metadata()
        if "resample" in self.plots:
            self.plot_trimmed_photos(self.plot_metadata[self.plot_metadata.flag == 0])
        if "polygon" in self.plots:
            self.plot_polygons()
        if self.plots == "resample-obscure":
            self.plot_brightness_hist(self.plot_metadata)

        self.logger.warning("Plots generated and saved on the output folder: %s", self.output_path)

    def plot_trimmed_photos(self, new_metadata: pd.DataFrame) -> None:
        """Plot the trimmed photos.

        Args:
            new_metadata (pd.DataFrame): The new metadata.
        """
        metadata = self.get_metadata()
        if "image-longitude" not in metadata.columns or "image-longitude" not in new_metadata.columns:
            self.logger.warning(
                "Longitude and Latitude columns are not found in the metadata.",
            )
            self.logger.warning("Plotting will not be performed.")
            return
        _, ax = plt.subplots(figsize=(20, 10))
        ax.plot(metadata["image-longitude"], metadata["image-latitude"], ".k")
        ax.plot(new_metadata["image-longitude"], new_metadata["image-latitude"], "xr")
        ax.legend(["Original", "After Sampling"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Comparison of Original and Samplingd Images")
        if self.metadata.dataset_metadata.get("trimmed_polygon") is not None:
            self.metadata.dataset_metadata["trimmed_polygon"].plot(ax=ax, color="none", edgecolor="black", linewidth=2)
        plt.savefig(self.output_path / "graph_trimmed_images.png")
        plt.close()

    def plot_polygons(self) -> None:
        """Plot the polygons."""
        metadata = self.get_metadata()
        gdf = gpd.GeoDataFrame(metadata, geometry="polygon_m")
        _, ax = plt.subplots(figsize=(15, 15))

        no_overlap = gdf[gdf.overlap == 0]
        overlap = gdf[gdf.overlap == 1]
        no_overlap.plot(ax=ax, facecolor="none", edgecolor="black", label="No Overlap")
        overlap.plot(ax=ax, facecolor="none", edgecolor="red", label="Overlap")
        if not no_overlap.empty or not overlap.empty:
            plt.legend()

        plt.title("Overlap of Images")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(self.output_path / "graph_polygons.png")
        plt.close()

    def plot_brightness_hist(self, metadata: pd.DataFrame) -> None:
        """Plot the images.

        Args:
            metadata (pd.DataFrame): The metadata with the images.
        """
        # metadata can have brightness column or brightness_1, brightness_2, brightness_3
        if "brightness" not in metadata.columns:
            brightness_columns = [col for col in metadata.columns if col.startswith("brightness_")]
            if len(brightness_columns) == 0:
                self.logger.warning("No brightness column found in the metadata.")
                return
            for col in brightness_columns:
                self._plot_individual_brightness(metadata, col)
        else:
            self._plot_individual_brightness(metadata, "brightness")

    def _plot_individual_brightness(self, metadata: pd.DataFrame, col: str) -> None:
        """Plot the individual brightness.

        Args:
            metadata (pd.DataFrame): The metadata with the images.
            col (str): The column name for brightness.
        """
        channel = col.split("_")
        channel = channel[-1] if len(channel) > 1 else "mean"
        plt.hist(metadata[col], bins=30, edgecolor="black")
        plt.xlabel("Mean Brightness")
        plt.ylabel("Frequency")
        plt.title("Distribution of Image Brightness - channel:" + channel)
        plt.savefig(self.output_path / f"graph_{col}_hist.png")
        plt.close()
