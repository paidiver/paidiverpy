"""Module to handle images and metadata for each step in the pipeline."""

import gc
import io
import logging
from pathlib import Path
import cv2
import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from IPython.display import HTML
from PIL import Image
from paidiverpy.config.configuration import Configuration
from paidiverpy.utils import formating_html
from paidiverpy.utils.data import NUM_CHANNELS_GREY
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.object_store import check_create_bucket_exists
from paidiverpy.utils.object_store import create_client
from paidiverpy.utils.object_store import upload_file_to_bucket


class ImagesLayer:
    """Class to handle images and metadata for each step in the pipeline.

    Args:
        output_path (str): Path to save the images. Default is None.
    """

    def __init__(self, output_path: str | None = None):
        self.steps = []
        self.step_metadata = []
        self.images = []
        self.max_images = 12
        self.output_path = output_path
        self.filenames = []

    def add_step(
        self,
        step: str,
        images: np.ndarray | da.core.Array = None,
        metadata: pd.DataFrame = None,
        step_metadata: dict | None = None,
        update_metadata: bool = False,
        track_changes: bool = True,
    ) -> None:
        """Add a step to the pipeline.

        Args:
            step (str): The step to add
            images (np.ndarray | da.core.Array, optional): The images to add.
        Defaults to None.
            metadata (pd.DataFrame, optional): The metadata to add. Defaults to None.
            step_metadata (dict, optional): The metadata for the step.
        Defaults to None.
            update_metadata (bool, optional): Whether to update the metadata.
            track_changes (bool, optional): Whether to track changes. Defaults to True.
        """
        if update_metadata:
            last_filenames = np.array(self.filenames[-1])
            new_filenames = np.isin(last_filenames, metadata["image-filename"])
            new_images = [image for image, filename in zip(self.images[-1], new_filenames, strict=False) if filename]
            if not track_changes and len(self.images) > 1:
                len_images = len(self.images[-1])
                del self.images[-1]
                self.images.append([None] * len_images)
            self.images.append(new_images)
        else:
            if not track_changes and len(self.images) > 0:
                len_images = len(self.images[-1])
                del self.images[-1]
                self.images.append([None] * len_images)
            self.images.append(images)
        self.step_metadata.append(step_metadata)
        self.steps.append(step)
        self.filenames.append(metadata["image-filename"].tolist())
        gc.collect()

    def remove_steps_by_order(self, step_order: int) -> None:
        """Remove steps by order.

        Args:
            step_order (int): The step order to remove
        """
        self.steps = self.steps[:step_order]
        self.images = self.images[:step_order]
        self.filenames = self.filenames[:step_order]

    def get_step(
        self,
        step: str | int | None = None,
        last: bool = False,
    ) -> list[np.ndarray | da.core.Array]:
        """Get a step by name or order.

        Args:
            step (str | int, optional): The step to get. Defaults to None.
            last (bool, optional): If True, get the last step. Defaults to False.

        Returns:
            list[np.ndarray | da.core.Array]: The images for the step
        """
        if last:
            return self.images[-1]
        return self.images[step]

    def show(self, image_number: int = 0) -> HTML:
        """Show the images in the pipeline.

        Args:
            image_number (int, optional): The index of the image to show. Defaults to 0.

        Returns:
            HTML: The HTML representation of the images
        """
        return formating_html.images_repr(self, image_number=image_number, html=True)

    def save(
        self,
        step: str | int | None = None,
        last: bool = True,
        output_path: str | None = None,
        image_format: str = "png",
        config: Configuration | None = None,
        metadata: pd.DataFrame | None = None,
        client: Client = None,
        n_jobs: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Save the images in the pipeline.

        Args:
            step (str| int, optional): The step to save. Defaults to None.
            last (bool, optional): If True, save the last step. Defaults to False.
            output_path (str, optional): The output path to save the images. Defaults to None.
            image_format (str, optional): The image format to save. Defaults to "png".
            config (Configuration, optional): The configuration object. Defaults to None.
            metadata (pd.DataFrame, optional): The metadata object. Defaults to None.
            client (Client, optional): The Dask client. Defaults to None.
            n_jobs (int, optional): The number of jobs to use. Defaults to 1.
            logger (logging.Logger, optional): The logger to log messages. Defaults to None.
        """
        images = self.get_step(step, last)
        output_path, is_remote = config.get_output_path(output_path)
        logger = logger if logger else initialise_logging()
        step_order = len(self.steps) - 1 if last else step
        # if metadata is not None:
        #     metadata = metadata[metadata["flag"] == 0] if last else metadata[(metadata["flag"] == 0) | (metadata["flag"] > step)]
        if is_remote:
            self.save_remote(images, output_path, image_format, metadata, client, n_jobs, step_order, logger)
        else:
            self.save_local(images, output_path, image_format, metadata, client, n_jobs, step_order, logger)

    def save_remote(
        self,
        images: list[np.ndarray | da.core.Array],
        output_path: str,
        image_format: str,
        metadata: pd.DataFrame,
        client: Client,
        n_jobs: int,
        step_order: int,
        logger: logging.Logger,
    ) -> None:
        """Save the images to a remote location.

        Args:
            images (list): The images to save.
            output_path (str): The output path to save the images.
            image_format (str): The image format to save.
            metadata (pd.DataFrame): The metadata object.
            client (Client): The Dask client.
            n_jobs (int): The number of jobs to use.
            step_order (int): The step order.
            logger (logging.Logger): The logger to log messages.
        """
        s3_client = create_client()
        bucket_name = output_path[5:].split("/")[0]
        check_create_bucket_exists(bucket_name, s3_client, logger)
        if client:
            logger.info("Uploading images to S3 using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path + f"{self.filenames[step_order][idx]}", image_format, s3_client, metadata)
                for idx, image in enumerate(images)
            ]
            with ProgressBar():
                futures = client.compute(delayed_tasks)
                client.gather(futures)
        elif n_jobs > 1:
            logger.info("Uploading images to S3 using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path + f"{self.filenames[step_order][idx]}", image_format, s3_client, metadata)
                for idx, image in enumerate(images)
            ]
            with dask.config.set(scheduler="threads", num_workers=n_jobs), ProgressBar():
                dask.compute(*delayed_tasks)
        else:
            logger.info("Uploading images to S3")
            for idx, image in enumerate(images):
                img_path = output_path + f"{self.filenames[step_order][idx]}"
                self.process_and_upload(image, img_path, image_format, s3_client, metadata)

    def save_local(
        self,
        images: list[np.ndarray | da.core.Array],
        output_path: str,
        image_format: str,
        metadata: pd.DataFrame,
        client: Client,
        n_jobs: int,
        step_order: int,
        logger: logging.Logger,
    ) -> None:
        """Save the images to a local location.

        Args:
            images (list): The images to save.
            output_path (str): The output path to save the images.
            image_format (str): The image format to save.
            metadata (pd.DataFrame): The metadata object.
            client (Client): The Dask client.
            n_jobs (int): The number of jobs to use.
            step_order (int): The step order.
            logger (logging.Logger): The logger to log messages.
        """
        if client:
            logger.info("Saving images using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path / f"{self.filenames[step_order][idx]}", image_format, None, metadata)
                for idx, image in enumerate(images)
            ]
            with ProgressBar():
                futures = client.compute(delayed_tasks)
                client.gather(futures)
        elif n_jobs > 1:
            logger.info("Saving images using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path / f"{self.filenames[step_order][idx]}", image_format, None, metadata)
                for idx, image in enumerate(images)
            ]
            with dask.config.set(scheduler="threads", num_workers=n_jobs), ProgressBar():
                dask.compute(*delayed_tasks)
        else:
            logger.info("Saving images")
            for idx, image in enumerate(images):
                # remove extension from name if exist and then add the new extension
                img_path = output_path / f"{self.filenames[step_order][idx]}"
                self.process_and_upload(image, img_path, image_format, None, metadata)

    def process_and_upload(
        self,
        image: np.ndarray | da.core.Array,
        img_path: str | Path,
        image_format: str,
        s3_client: Client | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> None:
        """Process and upload the images.

        Args:
            image (np.ndarray | da.core.Array): The image to process and upload.
            img_path (str | Path): The image path to save.
            image_format (str): The image format to save.
            s3_client (boto3.client, optional): The S3 client. Defaults to None.
            metadata (pd.DataFrame, optional): The metadata object. Defaults to None.
        """
        _ = metadata
        saved_image, _ = self.calculate_image(image)
        img_path_with_suffix = img_path.with_suffix(f".{image_format}")
        # exif_dict = MetadataParser.metadata_to_exif(str(img_path).split("/")[-1], metadata, image_format)
        if saved_image.dtype == np.uint16:
            if image_format.lower() in ["tiff", "png"]:
                if s3_client:
                    buffer = io.BytesIO()
                    img_pil = Image.fromarray(saved_image)
                    img_pil.save(buffer, format=image_format.upper())
                    buffer.seek(0)
                    upload_file_to_bucket(buffer, img_path_with_suffix, s3_client)
                else:
                    cv2.imwrite(img_path_with_suffix, saved_image)
            else:
                msg = f"16-bit images can only be saved as TIFF or PNG, not {image_format}"
                raise ValueError(msg)

        elif saved_image.dtype in [np.uint8, np.float32]:
            if s3_client:
                _, encoded_image = cv2.imencode(f".{image_format}", saved_image)
                buffer = io.BytesIO(encoded_image.tobytes())
                upload_file_to_bucket(buffer, img_path_with_suffix, s3_client)
            else:
                cv2.imwrite(img_path_with_suffix, saved_image)
                # plt.imsave(img_path_with_suffix, saved_image, cmap=cmap, format=image_format)

        else:
            msg = f"Unsupported image dtype: {saved_image.dtype}. Expected uint8, uint16, or float32."
            raise ValueError(msg)

    def calculate_image(self, image: np.ndarray | da.core.Array) -> tuple:
        """Calculate the image.

        Args:
            image (np.ndarray | da.core.Array): The image to calculate.

        Returns:
            tuple[np.ndarray, str]: The saved image and the colormap.
        """
        if len(image.shape) == NUM_DIMENSIONS_GREY:
            cmap = "gray"
            saved_image = image
        elif image.shape[-1] == NUM_CHANNELS_GREY:
            saved_image = np.squeeze(image, axis=-1)
            cmap = "gray"
        else:
            saved_image = image
            cmap = None
        return saved_image, cmap

    def remove(self, output_path: str | None = None) -> None:
        """Remove the images from the output path.

        Args:
            output_path (str, optional): The output path to save the images. Defaults to None.
        """
        is_docker = is_running_in_docker()
        if not output_path:
            output_path = self.output_path
        output_path = Path("/app/output/") if is_docker else output_path
        if output_path.exists():
            for file in output_path.iterdir():
                file.unlink()

    def __repr__(self) -> str:
        """Return the string representation of the object.

        Returns:
            str: The string representation of the object
        """
        repr_str = ""
        for step, image_paths in zip(self.steps, self.images, strict=False):
            repr_str += f"Step: {step}\n"
            for image_path in image_paths:
                repr_str += f"Image: {image_path}\n"
        return repr_str

    def _repr_html_(self) -> str:
        """Return the HTML representation of the object.

        Returns:
            str: The HTML representation of the object
        """
        return formating_html.images_repr(self, self.max_images)

    def __call__(self, max_images: int | None = None) -> HTML:
        """Call the object.

        Args:
            max_images (int, optional): The maximum number of images to show.
        Defaults to None.

        Returns:
            HTML: The HTML representation of the object
        """
        if not max_images:
            max_images = self.max_images
        return formating_html.images_repr(self, max_images=max_images, html=True)
