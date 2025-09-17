"""Module to handle images and metadata for each step in the pipeline."""

import gc
import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
import cv2
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from IPython.display import HTML
from PIL import Image
from paidiverpy.config.configuration import Configuration
from paidiverpy.utils import formating_html
from paidiverpy.utils.data import NUM_CHANNELS_GREY
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.object_store import check_create_bucket_exists
from paidiverpy.utils.object_store import create_client
from paidiverpy.utils.object_store import upload_file_to_bucket

logger = logging.getLogger("paidiverpy")


class ImagesLayer:
    """Class to handle images and metadata for each step in the pipeline.

    Args:
        output_path (str | Path | None): Path to save the images. Default is None.
    """

    def __init__(self, output_path: str | Path | None = None):
        self.steps: list[str] = []
        self.step_metadata: list[dict[str, object]] = []
        self.max_images = 12
        self.output_path = output_path
        self.images = xr.Dataset()

    def add_step(
        self,
        step: str,
        images: xr.Dataset,
        step_metadata: dict[str, object],
        metadata: pd.DataFrame | None = None,
        track_changes: bool = True,
    ) -> None:
        """Add a step to the pipeline.

        Args:
            step (str): The step to add
            images (xr.Dataset): The images for the step.
            step_metadata (dict): The metadata for the step.
            metadata (pd.DataFrame | None, optional): The metadata to set. Defaults to None.
            track_changes (bool, optional): Whether to track changes. Defaults to True.
        """
        self.step_metadata.append(step_metadata)
        self.steps.append(step)
        images = images.rename(
            {
                "images": f"images_{len(self.steps) - 1}",
                "x": f"x_{len(self.steps) - 1}",
                "y": f"y_{len(self.steps) - 1}",
                "band": f"band_{len(self.steps) - 1}",
                "original_height": f"original_height_{len(self.steps) - 1}",
                "original_width": f"original_width_{len(self.steps) - 1}",
            }
        )
        if not self.images or not track_changes:
            self.images = images
        else:
            self.images = xr.merge([self.images, images], join="outer")

        if metadata is not None:
            self.images = self.images.assign_coords(flag=("filename", metadata["flag"].values))

        gc.collect()

    def replace_step(self, images: xr.Dataset) -> None:
        """Add a step to the pipeline.

        Args:
            images (xr.Dataset): The images for the step.
        """
        step = len(self.steps) - 1
        images = images.rename(
            {
                "images": f"images_{step}",
                "x": f"x_{step}",
                "y": f"y_{step}",
                "band": f"band_{step}",
                "original_height": f"original_height_{step}",
                "original_width": f"original_width_{step}",
            }
        )
        variables = [var for var in self.images.data_vars if var.endswith(f"_{step}")]
        self.images = self.images.drop_vars(variables)
        dims = [dim for dim in self.images.dims if dim.endswith(f"_{step}")]
        self.images = self.images.drop_dims(dims)
        self.images = xr.merge([self.images, images], join="outer")
        gc.collect()

    def set_images(self, images: xr.Dataset) -> None:
        """Set the images for the layer.

        Args:
            images (xr.Dataset): The images to set.
        """
        self.images = images

    def remove_steps_by_order(self, step_order: int) -> None:
        """Remove steps by order.

        Args:
            step_order (int): The step order to remove
        """
        if not self.images:
            return
        steps_to_remove = range(step_order, len(self.steps))
        self.steps = self.steps[:step_order]
        self.step_metadata = self.step_metadata[:step_order]
        for step in steps_to_remove:
            variables = [var for var in self.images.data_vars if var.endswith(f"_{step}")]
            self.images = self.images.drop_vars(variables)
            coordinates = [coord for coord in self.images.coords if coord.endswith(f"_{step}")]
            self.images = self.images.drop_vars(coordinates)
            dims = [dim for dim in self.images.dims if dim.endswith(f"_{step}")]
            self.images = self.images.drop_dims(dims)
        flags = self.images.flag.where(self.images.flag <= step_order, 0)
        self.images = self.images.assign_coords(flag=("filename", flags.to_numpy()))

    def get_step(self, step: str | int | None = None, last: bool = False, flag: None | int = None) -> xr.Dataset:
        """Get a step by name or order.

        Args:
            step (str | int, optional): The step to get. Defaults to None.
            last (bool, optional): If True, get the last step. Defaults to False.
            flag (None | int, optional): The flag to filter the images. Defaults to None.

        Returns:
            xr.Dataset | None: The images for the step or None if the step does not exist.
        """
        if last:
            step = len(self.steps) - 1
        variable_name = f"images_{step}"
        # check if variable_name exists
        if variable_name not in self.images.data_vars:
            return xr.Dataset()
        images = self.images[[variable_name]]
        if flag is not None:
            image_dtype = images[variable_name].dtype
            images = images.where(images["flag"] == flag, drop=True)
            images[variable_name] = images[variable_name].astype(image_dtype)
        return images.rename(
            {
                variable_name: "images",
                f"y_{step}": "y",
                f"x_{step}": "x",
                f"band_{step}": "band",
                f"original_height_{step}": "original_height",
                f"original_width_{step}": "original_width",
            }
        )

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
        config: Configuration,
        step: str | int | None = None,
        last: bool = True,
        output_path: str | Path | None = None,
        image_format: str = "png",
        client: Client | None = None,
        n_jobs: int = 1,
        use_dask: bool = False,
    ) -> None:
        """Save the images in the pipeline.

        Args:
            step (str| int, optional): The step to save. Defaults to None.
            last (bool, optional): If True, save the last step. Defaults to False.
            output_path (str, optional): The output path to save the images. Defaults to None.
            image_format (str, optional): The image format to save. Defaults to "png".
            config (Configuration, optional): The configuration object. Defaults to None.
            client (Client, optional): The Dask client. Defaults to None.
            n_jobs (int, optional): The number of jobs to use. Defaults to 1.
            use_dask (bool, optional): Whether to use Dask. Defaults to False.
        """
        images = self.get_step(step, last)
        output_path, is_remote = config.get_output_path(output_path)
        s3_client = None
        if is_remote:
            s3_client = create_client()
            bucket_name = str(output_path)[5:].split("/")[0]
            check_create_bucket_exists(bucket_name, s3_client)
        tasks = xr.apply_ufunc(
            ImagesLayer.process_single_image,
            images["images"],
            images["original_height"],
            images["original_width"],
            images["filename"],
            kwargs={
                "output_path": output_path,
                "image_format": image_format,
                "s3_client": s3_client,
                "processor": self.process_and_upload,
            },
            input_core_dims=[["y", "x", "band"], [], [], []],
            vectorize=True,
            dask="parallelized" if use_dask else "forbidden",
            output_dtypes=[int],
        )

        if use_dask:
            if client:
                logger.info("Saving images using Dask")

                futures = client.compute(tasks)
                with ProgressBar():
                    client.gather(futures)
            else:
                logger.info("Saving images using Threads")
                with dask.config.set(scheduler="threads", num_workers=n_jobs), ProgressBar():
                    tasks.compute()

    def process_and_upload(
        self,
        image: np.ndarray[Any, Any] | da.core.Array,
        img_path: str | Path,
        image_format: str,
        s3_client: Client | None = None,
    ) -> None:
        """Process and upload the images.

        Args:
            image (np.ndarray | da.core.Array): The image to process and upload.
            img_path (str | Path): The image path to save.
            image_format (str): The image format to save.
            s3_client (boto3.client, optional): The S3 client. Defaults to None.
        """
        saved_image = self.calculate_image(image)
        img_path_with_suffix = img_path.with_suffix(f".{image_format}") if isinstance(img_path, Path) else f"{img_path}.{image_format}"
        if saved_image.dtype == np.uint16:
            if image_format.lower() in ["tiff", "png"]:
                if s3_client:
                    buffer = io.BytesIO()
                    img_pil = Image.fromarray(saved_image)
                    img_pil.save(buffer, format=image_format.upper())
                    buffer.seek(0)
                    upload_file_to_bucket(buffer, str(img_path_with_suffix), s3_client)
                else:
                    cv2.imwrite(str(img_path_with_suffix), saved_image)
            else:
                msg = f"16-bit images can only be saved as TIFF or PNG, not {image_format}"
                raise ValueError(msg)

        elif saved_image.dtype in [np.uint8, np.float32]:
            if s3_client:
                _, encoded_image = cv2.imencode(f".{image_format}", saved_image)
                buffer = io.BytesIO(encoded_image.tobytes())
                upload_file_to_bucket(buffer, str(img_path_with_suffix), s3_client)
            else:
                cv2.imwrite(str(img_path_with_suffix), saved_image)
                # plt.imsave(img_path_with_suffix, saved_image, cmap=cmap, format=image_format)

        else:
            msg = f"Unsupported image dtype: {saved_image.dtype}. Expected uint8, uint16, or float32."
            raise ValueError(msg)

    def calculate_image(self, image: np.ndarray[Any, Any] | da.core.Array) -> np.ndarray[Any, Any]:
        """Calculate the image.

        Args:
            image (np.ndarray | da.core.Array): The image to calculate.

        Returns:
            np.ndarray: The calculated image.
        """
        if len(image.shape) == NUM_DIMENSIONS_GREY:
            saved_image = image
        elif image.shape[-1] == NUM_CHANNELS_GREY:
            saved_image = np.squeeze(image, axis=-1)
        else:
            saved_image = image
        return saved_image

    def remove(self, output_path: str | Path | None = None) -> None:
        """Remove the images from the output path.

        Args:
            output_path (str | Path, optional): The output path to save the images. Defaults to None.
        """
        is_docker = is_running_in_docker()
        if not output_path:
            output_path = self.output_path
        output_path = Path("/app/output/") if is_docker else output_path
        if output_path.exists():
            for file in output_path.iterdir():
                file.unlink()

    def _repr_html_(self) -> str:
        """Return the HTML representation of the object.

        Returns:
            str: The HTML representation of the object
        """
        return formating_html.images_repr(self, max_images=self.max_images)

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

    @staticmethod
    def process_single_image(
        img: np.ndarray[Any, Any],
        height: np.ndarray[Any, Any],
        width: np.ndarray[Any, Any],
        filename: np.ndarray[Any, Any],
        output_path: Path,
        image_format: str,
        s3_client: Client | None,
        processor: Callable,
    ) -> int:
        """Process a single image and save it.

        Args:
            img (np.ndarray): The image to process.
            height (np.ndarray): The height of the image.
            width (np.ndarray): The width of the image.
            filename (np.ndarray): The filename of the image.
            output_path (Path): The path to save the output.
            image_format (str): The format to save the image.
            s3_client (Client | None): The S3 client to use for uploading.
            processor (Callable): The processing function to use.

        Returns:
            int: The status code (0 for success).
        """
        cropped = img[:height, :width, :]
        output_path = output_path + filename.item() if s3_client else output_path / filename.item()
        processor(cropped, output_path, image_format, s3_client)
        return 0
