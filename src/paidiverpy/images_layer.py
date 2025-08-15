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


class ImagesLayer:
    """Class to handle images and metadata for each step in the pipeline.

    Args:
        output_path (str): Path to save the images. Default is None.
    """

    def __init__(self, output_path: str | None = None):
        self.steps = []
        self.step_metadata = []
        self.max_images = 12
        self.output_path = output_path
        self.images = None

    def add_step(
        self,
        step: str,
        images: xr.Dataset,
        step_metadata: dict | None = None,
        metadata: pd.DataFrame | None = None,
        track_changes: bool = True,
    ) -> None:
        """Add a step to the pipeline.

        Args:
            step (str): The step to add
            images (xr.Dataset): The images for the step.
            step_metadata (dict, optional): The metadata for the step.
        Defaults to None.
            metadata (pd.DataFrame, optional): The metadata for the step.
        Defaults to None.
            track_changes (bool, optional): Whether to track changes. Defaults to True.
        """
        if not metadata:
            metadata = pd.DataFrame()
        self.step_metadata.append(step_metadata)
        self.steps.append(step)
        images = images.rename(
            {
                "image": f"image_{len(self.steps) - 1}",
                "mask": f"mask_{len(self.steps) - 1}",
                "x": f"x_{len(self.steps) - 1}",
                "y": f"y_{len(self.steps) - 1}",
                "band": f"band_{len(self.steps) - 1}",
            }
        )
        if self.images is None or not track_changes:
            self.images = images
        else:
            self.images = xr.concat([self.images, images], dim="filename")

        if len(metadata.columns) > 0:
            self.update_metadata_coordinates(metadata)

        gc.collect()

    def update_metadata_coordinates(self, metadata: pd.DataFrame) -> None:
        """Update the metadata coordinates.

        Args:
            metadata (pd.DataFrame): The metadata to update.
        """
        for col in metadata.columns:
            if col in self.images.coords:
                updated_values = self.images[col].to_series()
                for fname in metadata.index:
                    if fname in updated_values.index:
                        updated_values.loc[fname] = metadata.loc[fname, col]
                self.images = self.images.assign_coords({col: ("filename", updated_values.to_numpy())})
            else:
                values = []
                for fname in self.images["filename"].to_numpy():
                    if fname in metadata.index:
                        values.append(metadata.loc[fname, col])
                    else:
                        values.append(None)
                self.images = self.images.assign_coords({col: ("filename", values)})

    def replace_step(self, images: xr.Dataset, metadata: pd.DataFrame | None = None) -> None:
        """Add a step to the pipeline.

        Args:
            step (str): The step to add
            images (xr.Dataset): The images for the step.
            metadata (pd.DataFrame | None): The metadata for the step.
        """
        if not metadata:
            metadata = pd.DataFrame()
        images = images.rename(
            {
                "image": f"image_{len(self.steps) - 1}",
                "mask": f"mask_{len(self.steps) - 1}",
                "x": f"x_{len(self.steps) - 1}",
                "y": f"y_{len(self.steps) - 1}",
                "band": f"band_{len(self.steps) - 1}",
            }
        )
        self.images = xr.concat([self.images, images], dim="filename")
        if len(metadata.columns) > 0:
            self.update_metadata_coordinates(metadata)

        gc.collect()

    def remove_steps_by_order(self, step_order: int) -> None:
        """Remove steps by order.

        Args:
            step_order (int): The step order to remove
        """
        self.steps = self.steps[:step_order]
        self.step_metadata = self.step_metadata[:step_order]
        variables = [var for var in self.images.data_vars if var.endswith(f"_{step_order - 1}")]
        self.images = self.images.drop_vars(variables)
        self.images.flag = self.images.flag.where(self.images.flag <= step_order, 0)

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
            step = len(self.steps) - 1
        images = self.images[[f"image_{step}", f"mask_{step}"]]
        return images.rename({f"image_{step}": "image", f"mask_{step}": "mask", f"y_{step}": "y", f"x_{step}": "x", f"band_{step}": "band"})

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
        client: Client = None,
        n_jobs: int = 1,
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
        """
        images = self.get_step(step, last)
        output_path, is_remote = config.get_output_path(output_path)
        s3_client = None
        if is_remote:
            s3_client = create_client()
            bucket_name = output_path[5:].split("/")[0]
            check_create_bucket_exists(bucket_name, s3_client)
        tasks = xr.apply_ufunc(
            ImagesLayer.process_single_image,
            images["image"],
            images["mask"],
            images["filename"],
            kwargs={
                "output_path": output_path,
                "image_format": image_format,
                "s3_client": s3_client,
                "processor": self.process_and_upload,
            },
            input_core_dims=[["y", "x", "band"], [], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        )

        if client:
            logging.info("Saving images using Dask")

            futures = client.compute(tasks)
            with ProgressBar():
                client.gather(futures)
        else:
            logging.info("Saving images using Threads")
            with dask.config.set(scheduler="threads", num_workers=n_jobs), ProgressBar():
                tasks.compute()

    def process_and_upload(
        self,
        image: np.ndarray | da.core.Array,
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

    # TODO: update this method to use the new format
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

    @staticmethod
    def process_single_image(
        img: np.ndarray, mask: np.ndarray, filename: str, output_path: Path, image_format: str, s3_client: Client | None, processor: callable
    ) -> int:
        """Process a single image and save it.

        Args:
            img (np.ndarray): The image to process.
            mask (np.ndarray): The mask to apply.
            filename (str): The name of the file to save.
            output_path (Path): The path to save the output.
            image_format (str): The format to save the image.
            s3_client (Client | None): The S3 client to use for uploading.
            processor (callable): The processing function to use.

        Returns:
            int: The status code (0 for success).
        """
        img_masked = img.copy()
        img_masked[mask == 0] = 0
        processor(img_masked, output_path / filename.item(), image_format, s3_client)
        return 0
