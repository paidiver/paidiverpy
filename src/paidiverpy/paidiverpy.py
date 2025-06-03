"""Main class for the paidiverpy package."""

import logging
from contextlib import suppress
from functools import partial
from pathlib import Path
import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from distributed import LocalCluster
from tqdm import tqdm
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser
from paidiverpy.utils.base_model import BaseModel
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.parallellisation import get_client
from paidiverpy.utils.parallellisation import get_n_jobs


class Paidiverpy:
    """Main class for the paidiverpy package.

    Args:
        config_params (dict | ConfigParams, optional): The configuration parameters.
            It can contain the following keys / attributes:
            - input_path (str): The path to the input files.
            - output_path (str): The path to the output files.
            - image_open_args (str): The type of the images.
            - metadata_path (str): The path to the metadata file.
            - metadata_type (str): The type of the metadata file.
            - track_changes (bool): Whether to track changes.
            - n_jobs (int): The number of n_jobs.
        config_file_path (str, optional): The path to the configuration file.
        config (Configuration, optional): The configuration object.
        metadata (MetadataParser, optional): The metadata object.
        images (ImagesLayer, optional): The images object.
        client (Client, optional): The Dask client object.
        paidiverpy (Paidiverpy, optional): The paidiverpy object.
        track_changes (bool): Whether to track changes. Defaults to None, which means
            it will be set to the value of the configuration file.
        logger (logging.Logger, optional): The logger object.
        raise_error (bool, optional): Whether to raise an error.
        verbose (int, optional): verbose level (0 = none, 1 = errors/warnings, 2 = info).
    """

    def __init__(
        self,
        config_params: dict | ConfigParams = None,
        config_file_path: str | None = None,
        config: Configuration = None,
        metadata: MetadataParser = None,
        images: ImagesLayer = None,
        client: Client | None = None,
        paidiverpy: "Paidiverpy" = None,
        track_changes: bool | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        if paidiverpy:
            self._set_variables_from_paidiverpy(paidiverpy)
        else:
            self.raise_error = raise_error
            self.verbose = verbose
            self.logger = logger or initialise_logging(verbose=self.verbose)
            try:
                self.config = config or self._initialise_config(config_file_path, config_params)
            except Exception as error:
                msg = f"{error}"
                self.logger.error(msg)
                raise
            self.metadata = metadata or self._initialize_metadata()
            self.images = images or ImagesLayer(
                output_path=self.config.general.output_path,
            )
            if not client:
                result = get_client(self.config.general.client, self.config.general.n_jobs)
                if isinstance(result, tuple):
                    self.client, self.job_id = result
                else:
                    self.client = result
                    self.job_id = None
            else:
                self.client = client
                self.job_id = None
            self.n_jobs = get_n_jobs(self.config.general.n_jobs)
            self.track_changes = self.config.general.track_changes
        self.track_changes = self.track_changes if track_changes is None else track_changes
        self.layer_methods = None

    def run(self, add_new_step: bool = True) -> ImagesLayer | None:
        """Run the paidiverpy pipeline.

        Args:
            add_new_step (bool, optional): Whether to add a new step. Defaults to True.

        Returns:
            ImagesLayer | None: The images object.
        """
        mode = self.step_metadata.get("mode")
        test = self.step_metadata.get("test")
        params = self.step_metadata.get("params") or {}
        method, params = self._get_method_by_mode(params, self.layer_methods, mode)
        images = self.images.get_step(step=len(self.images.images) - 1)
        image_list, metadata = (
            self.process_sequentially(images, method, params) if self.n_jobs == 1 else self.process_parallel(images, method, params)
        )
        if not test:
            self.step_name = f"step_{self.config_index}" if not self.step_name else self.step_name
            self.set_metadata(metadata, flag=len(self.images.images))
            if add_new_step:
                self.images.add_step(
                    step=self.step_name,
                    images=image_list,
                    step_metadata=self.step_metadata,
                    metadata=self.get_metadata(),
                    track_changes=self.track_changes,
                )
                return None
            self.images.images[-1] = image_list
            return self.images
        return None

    def process_sequentially(
        self, images: list[np.ndarray], method: callable, params: dict, custom: bool = False
    ) -> tuple[list[np.ndarray], pd.DataFrame]:
        """Process the images sequentially.

        Method to process the images sequentially.

        Args:
            images (List[np.ndarray]): The list of images to process.
            method (callable): The method to apply to the images.
            params (dict): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            tuple[list[np.ndarray], pd.DataFrame]: A tuple containing the list of processed images and the metadata DataFrame.
        """
        func = partial(method, params=params)
        metadata = self.get_metadata().to_dict(orient="records")
        processed_images = []
        for index, (img, metadata_image) in enumerate(tqdm(zip(images, metadata, strict=False), total=len(images), desc="Processing images")):
            if custom:
                image, metadata_image_updated = func(img, metadata=metadata_image, metadata_core=metadata).process()
            else:
                image, metadata_image_updated = func(img, metadata=metadata_image, metadata_core=metadata)

            metadata[index] = metadata_image_updated
            processed_images.append(image)

        metadata = pd.DataFrame(metadata)

        return processed_images, metadata

    def process_parallel(
        self,
        images: list[da.core.Array],
        method: callable,
        params: BaseModel,
        custom: bool = False,
    ) -> tuple[list[np.ndarray], pd.DataFrame]:
        """Process the images in parallel.

        Method to process the images in parallel.

        Args:
            images (List[da.core.Array]): The list of images to process.
            method (callable): The method to apply to the images.
            params (BaseModel): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            tuple[list[np.ndarray], pd.DataFrame]: A tuple containing the list of processed images and the metadata DataFrame.
        """
        func = partial(method, params=params)
        metadata = self.get_metadata().to_dict(orient="records")

        def _process_image(img, metadata_image, metadata_core):  # noqa: ANN001, ANN202
            if custom:
                result = func(img, metadata=metadata_image, metadata_core=metadata_core).process()
            else:
                result = func(img, metadata=metadata_image, metadata_core=metadata_core)
            return result

        tasks = [dask.delayed(_process_image)(img, metadata_image, metadata) for img, metadata_image in zip(images, metadata, strict=False)]

        if self.client:
            if isinstance(self.client.cluster, LocalCluster):
                futures = self.client.compute(tasks)
            else:
                futures = [
                    self.client.submit(_process_image, img, metadata_image, metadata) for img, metadata_image in zip(images, metadata, strict=False)
                ]
            with ProgressBar():
                results = self.client.gather(futures)
        else:
            with dask.config.set(scheduler="threads", num_workers=self.n_jobs), ProgressBar():
                results = dask.compute(*tasks)

        processed_images, metadata = zip(*results, strict=False)
        metadata = pd.DataFrame(metadata)

        return list(processed_images), metadata

    def process_dataset(
        self,
        images: list[da.core.Array],
        method: callable,
        params: BaseModel,
        custom: bool = False,
    ) -> tuple[list[np.ndarray], pd.DataFrame]:
        """Process the images as a dataset.

        Args:
            images (List[da.core.Array]): The list of images to process.
            method (callable): The method to apply to the images.
            params (BaseModel): The parameters for the method.
            custom (bool, optional): Whether the method is a custom method. Defaults to False.

        Returns:
            tuple[list[np.ndarray], pd.DataFrame]: A tuple containing the list of processed images and the metadata DataFrame.
        """
        func = partial(method, params=params)
        metadata = self.get_metadata().to_dict(orient="records")
        if custom:
            processed_images, metadata = func(images, metadata=metadata).process()
        else:
            processed_images, metadata = func(images, metadata=metadata)

        metadata = pd.DataFrame(metadata)

        return processed_images, metadata

    def _set_variables_from_paidiverpy(self, paidiverpy: "Paidiverpy") -> None:
        """Set the variables from the paidiverpy object.

        Args:
            paidiverpy (Paidiverpy): The paidiverpy object.
        """
        self.logger = paidiverpy.logger
        self.images = paidiverpy.images
        self.config = paidiverpy.config
        self.metadata = paidiverpy.metadata
        self.verbose = paidiverpy.verbose
        self.raise_error = paidiverpy.raise_error
        self.n_jobs = paidiverpy.n_jobs
        self.track_changes = paidiverpy.track_changes
        self.client = paidiverpy.client
        self.job_id = paidiverpy.job_id

    def _initialise_config(
        self,
        config_file_path: str,
        config_params: ConfigParams | dict,
    ) -> Configuration:
        """Initialize the configuration object.

        Args:
            config_file_path (str): Configuration file path.
            config_params (ConfigParams | dict): Configuration parameters.

        Returns:
            Configuration: The configuration object.
        """
        if config_file_path:
            return Configuration(config_file_path=config_file_path)
        general_config = {}
        config_params = ConfigParams(**config_params) if isinstance(config_params, dict) else config_params
        config_params_keys = ["input_path", "output_path", "metadata_path", "metadata_type", "image_open_args", "track_changes", "n_jobs"]
        for key in config_params_keys:
            general_config[key] = getattr(config_params, key)
        return Configuration(add_general=general_config)

    def _initialize_metadata(self) -> MetadataParser:
        """Initialize the metadata object.

        Returns:
            MetadataParser: The metadata object.
        """
        general = self.config.general
        if getattr(general, "metadata_path", None) and getattr(
            general,
            "metadata_type",
            None,
        ):
            return MetadataParser(config=self.config, logger=self.logger)
        self.logger.info(
            "Metadata type is not specified. Loading files from the input path.",
        )
        self.logger.info("Metadata will be created from the files in the input path.")
        input_path = Path(general.input_path)
        file_pattern = general.file_name_pattern
        list_of_files = list(input_path.glob(file_pattern))
        metadata = pd.DataFrame(list_of_files, columns=["image-filename"])
        return metadata.reset_index().rename(columns={"index": "ID"})

    def get_metadata(self, flag: int | None = None) -> pd.DataFrame:
        """Get the metadata object.

        Args:
            flag (int, optional): The flag value. Defaults to None.

        Returns:
            pd.DataFrame: The metadata object.
        """
        flag = 0 if flag is None else flag
        if flag == "all":
            if "image-datetime" not in self.metadata.metadata.columns:
                return self.metadata.metadata.copy()
            return self.metadata.metadata.sort_values("image-datetime").copy()
        if "image-datetime" not in self.metadata.metadata.columns:
            return self.metadata.metadata[self.metadata.metadata["flag"] <= flag].copy()
        return self.metadata.metadata[self.metadata.metadata["flag"] <= flag].sort_values("image-datetime").copy()

    def set_metadata(self, metadata: pd.DataFrame, flag: bool = False) -> None:
        """Set the metadata.

        Args:
            metadata (pd.DataFrame): The metadata object.
            flag (bool, optional): The flag value. Defaults to False.
        """
        if not flag:
            self.metadata.metadata = metadata
        else:
            original_metadata = self.metadata.metadata.copy()

            merged_metadata = original_metadata.merge(metadata, on="image-filename", how="left", suffixes=("_orig", "_new"))

            updated_metadata = merged_metadata[["image-filename"]].copy()

            all_columns = set(original_metadata.columns).union(metadata.columns) - {"image-filename"}

            for column in all_columns:
                col_new = f"{column}_new" if f"{column}_new" in merged_metadata.columns else None
                col_orig = f"{column}_orig" if f"{column}_orig" in merged_metadata.columns else column

                if col_new and col_orig:
                    updated_metadata[column] = merged_metadata[col_new].combine_first(merged_metadata[col_orig])
                elif col_new:
                    updated_metadata[column] = merged_metadata[col_new]
                elif col_orig:
                    updated_metadata[column] = merged_metadata[col_orig]

            updated_metadata = updated_metadata[
                ["image-filename"]
                + [col for col in original_metadata.columns if col != "image-filename"]
                + [col for col in updated_metadata.columns if col not in original_metadata.columns and col != "image-filename"]
            ]

            for col in original_metadata.columns:
                if col in updated_metadata.columns:
                    with suppress(Exception):
                        updated_metadata[col] = updated_metadata[col].astype(original_metadata[col].dtype)
            self.metadata.metadata = updated_metadata

    def save_images(
        self,
        step: str | int | None = None,
        image_format: str = "png",
        output_path: str | Path | None = None,
    ) -> None:
        """Save the images.

        Args:
            step (int, optional): The step order. Defaults to None.
            image_format (str, optional): The image format. Defaults to "png".
            output_path (str | Path, optional): The output path. Defaults to None.
        """
        last = False
        if step is None:
            last = True
        if not output_path:
            output_path = self.config.general.output_path
        self.logger.info("Saving images from step: %s", step if not last else "last")

        # metadata = MetadataParser.group_metadata_and_dataset_metadata(self.metadata.metadata,
        #                                                               self.metadata.dataset_metadata)
        self.images.save(
            step,
            last=last,
            output_path=output_path,
            image_format=image_format,
            config=self.config,
            metadata=None,
            client=self.client,
            n_jobs=self.n_jobs,
            logger=self.logger,
        )
        self.metadata.dataset_metadata["output_path"] = str(output_path)
        self.logger.info("Images are saved to: %s", output_path)

    def remove_images(self) -> None:
        """Remove output images from the output path."""
        output_path = self.config.general.output_path
        self.logger.info("Removing images from the output path: %s", output_path)
        self.images.remove(output_path)

    def clear_steps(self, value: int | str) -> None:
        """Clear steps from the images and metadata.

        Args:
            value (int | str): Step name or order.
        """
        self.images.remove_steps_by_order(value)
        metadata = self.get_metadata(flag="all")
        metadata.loc[metadata["flag"] >= value, "flag"] = 0
        self.set_metadata(metadata)

    def _calculate_steps_metadata(self, config_part: Configuration) -> dict:
        """Calculate the steps metadata.

        Args:
            config_part (Configuration): The configuration part.

        Returns:
            dict: The steps metadata.
        """
        return dict(config_part.__dict__.items())

    def _get_method_by_mode(
        self,
        params: BaseModel,
        method_dict: dict,
        mode: str,
        class_method: bool = True,
    ) -> tuple:
        """Get the method by mode.

        Args:
            params (BaseModel): The parameters.
            method_dict (dict): The method dictionary.
            mode (str): The mode.
            class_method (bool, optional): Whether the method is a class method.
                Defaults to True.

        Raises:
            ValueError: Unsupported mode.

        Returns:
            tuple: The method and parameters.
        """
        # if mode not in method_dict:
        #     msg = f"Unsupported mode: {mode}"
        #     raise ValueError(msg)
        method_info = method_dict[mode]
        if not isinstance(params, method_info["params"]):
            params = method_info["params"](**params)
        method_name = method_info["method"]
        method = getattr(self.__class__, method_name) if class_method else getattr(self, method_name)

        return method, params

    def _calculate_raise_error(self) -> bool:
        """Calculate whether to raise an error.

        Returns:
            bool: Whether to raise an error.
        """
        if self.raise_error:
            return self.raise_error
        if isinstance(self.step_metadata["params"], BaseModel):
            raise_error = self.step_metadata["params"].raise_error
        else:
            raise_error = self.step_metadata["params"].get("raise_error", False)
        return raise_error

    @staticmethod
    def prepare_inputs(
        image_data: np.ndarray,
        metadata: dict | None,
        params: BaseModel | None,
        default_params_factory: BaseModel,
        **kwargs: dict,
    ) -> tuple[np.ndarray, dict, BaseModel]:
        """Standard preprocessing for convert layer methods.

        Args:
            image_data (np.ndarray): The image data.
            metadata (dict | None): The metadata.
            params (BaseModel | None): The parameters.
            default_params_factory (BaseModel): The default parameters factory.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple[np.ndarray, dict, BaseModel]: The image data, metadata, and parameters.
        """
        _ = kwargs
        metadata = metadata or {}
        params = params or default_params_factory()
        return image_data, metadata, params, kwargs
