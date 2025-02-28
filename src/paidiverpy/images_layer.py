"""Module to handle images and metadata for each step in the pipeline."""

import base64
import gc
import io
import logging
from io import BytesIO
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
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.logging_functions import initialise_logging
from paidiverpy.utils.object_store import check_create_bucket_exists
from paidiverpy.utils.object_store import create_client
from paidiverpy.utils.object_store import upload_file_to_bucket

MAX_IMAGES_TO_SHOW = 12
NUM_CHANNELS_GRAY = 1
NUM_CHANNELS_RGBA = 4
NUM_DIMS_GRAY = 2


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

    def remove_steps_by_name(self, step: tuple) -> int:
        """Remove steps by name.

        Args:
            step (str): The step to remove

        Returns:
            int: The index of the removed step
        """
        index = self.steps.index(step)
        self.steps = self.steps[:index]
        self.images = self.images[:index]
        self.filenames = self.filenames[:index]
        return index

    def remove_steps_by_order(self, step_order: int) -> None:
        """Remove steps by order.

        Args:
            step_order (int): The step order to remove
        """
        self.steps = self.steps[:step_order]
        self.images = self.images[:step_order]
        self.filenames = self.filenames[:step_order]

    def get_last_step_order(self) -> int:
        """Get the last step order.

        Returns:
            int: The last step order
        """
        return len(self.steps) - 1

    def get_step(
        self,
        step: str | int | None = None,
        by_order: bool = False,
        last: bool = False,
    ) -> list[np.ndarray | da.core.Array]:
        """Get a step by name or order.

        Args:
            step (str | int, optional): The step to get. Defaults to None.
            by_order (bool, optional): If True, get the step by order. Defaults to False.
            last (bool, optional): If True, get the last step. Defaults to False.

        Returns:
            list[np.ndarray | da.core.Array]: The images for the step
        """
        if last:
            return self.images[-1]
        index = step if by_order else self.steps.index(step)
        return self.images[index]

    def show(self, image_number: int = 0) -> None:
        """Show the images in the pipeline.

        Args:
            image_number (int, optional): The index of the image to show. Defaults to 0.
        """
        return HTML(self._generate_html(image_number=image_number))

    def save(
        self,
        step: str | int | None = None,
        by_order: bool = False,
        last: bool = False,
        output_path: str | None = None,
        image_format: str = "png",
        client: Client = None,
        n_jobs: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Save the images in the pipeline.

        Args:
            step (str| int, optional): The step to save. Defaults to None.
            by_order (bool, optional): If True, save the step by order. Defaults to False.
            last (bool, optional): If True, save the last step. Defaults to False.
            output_path (str, optional): The output path to save the images. Defaults to None.
            image_format (str, optional): The image format to save. Defaults to "png".
            client (Client, optional): The Dask client. Defaults to None.
            n_jobs (int, optional): The number of jobs to use. Defaults to 1.
            logger (logging.Logger, optional): The logger to log messages. Defaults to None.
        """
        images = self.get_step(step, by_order, last)
        is_remote = str(output_path).startswith("s3://")
        if not logger:
            logger = initialise_logging()

        if last:
            step_order = len(self.steps) - 1
        elif by_order:
            step_order = step
        else:
            step_order = self.steps.index(step)
        if is_remote:
            self.save_remote(images, output_path, image_format, client, n_jobs, step_order, logger)
        else:
            self.save_local(images, output_path, image_format, client, n_jobs, step_order, logger)

    def save_remote(
        self,
        images: list[np.ndarray | da.core.Array],
        output_path: str,
        image_format: str,
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
                dask.delayed(self.process_and_upload)(image, output_path + f"{self.filenames[step_order][idx]}", image_format, s3_client)
                for idx, image in enumerate(images)
            ]
            with ProgressBar():
                futures = client.compute(delayed_tasks)
                client.gather(futures)
        elif n_jobs > 1:
            logger.info("Uploading images to S3 using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path + f"{self.filenames[step_order][idx]}", image_format, s3_client)
                for idx, image in enumerate(images)
            ]
            with dask.config.set(scheduler="threads", num_workers=n_jobs), ProgressBar():
                dask.compute(*delayed_tasks)
        else:
            logger.info("Uploading images to S3")
            for idx, image in enumerate(images):
                img_path = output_path + f"{self.filenames[step_order][idx]}"
                self.process_and_upload(image, img_path, image_format, s3_client)

    def save_local(
        self,
        images: list[np.ndarray | da.core.Array],
        output_path: str,
        image_format: str,
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
            client (Client): The Dask client.
            n_jobs (int): The number of jobs to use.
            step_order (int): The step order.
            logger (logging.Logger): The logger to log messages.
        """
        is_docker = is_running_in_docker()
        if is_docker:
            output_path = Path("/app/output/")
        if not output_path:
            output_path = self.output_path
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if client:
            logger.info("Saving images using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path / f"{self.filenames[step_order][idx]}", image_format)
                for idx, image in enumerate(images)
            ]
            with ProgressBar():
                futures = client.compute(delayed_tasks)
                client.gather(futures)
        elif n_jobs > 1:
            logger.info("Saving images using Dask")
            delayed_tasks = [
                dask.delayed(self.process_and_upload)(image, output_path / f"{self.filenames[step_order][idx]}", image_format)
                for idx, image in enumerate(images)
            ]
            with dask.config.set(scheduler="threads", num_workers=n_jobs), ProgressBar():
                dask.compute(*delayed_tasks)
        else:
            logger.info("Saving images")
            for idx, image in enumerate(images):
                # remove extension from name if exist and then add the new extension
                img_path = output_path / f"{self.filenames[step_order][idx]}"
                self.process_and_upload(image, img_path, image_format)

    def process_and_upload(self, image: np.ndarray | da.core.Array, img_path: str | Path, image_format: str, s3_client: Client | None = None) -> None:
        """Process and upload the images.

        Args:
            image (np.ndarray | da.core.Array): The image to process and upload.
            img_path (str | Path): The image path to save.
            image_format (str): The image format to save.
            s3_client (boto3.client, optional): The S3 client. Defaults to None.
        """
        saved_image, cmap = self.calculate_image(image)
        if isinstance(img_path, str):
            if "." in img_path:
                img_path = img_path.split(".")[0]
            img_path = f"{img_path}.{image_format.lower()}"
        else:
            img_path = img_path.with_suffix(f".{image_format}")
        if saved_image.dtype == np.uint16:
            if image_format.lower() in ["tiff", "png"]:
                if s3_client:
                    buffer = io.BytesIO()
                    img_pil = Image.fromarray(saved_image)
                    img_pil.save(buffer, format=image_format.upper())
                    buffer.seek(0)
                    upload_file_to_bucket(buffer, img_path, s3_client)
                else:
                    cv2.imwrite(img_path, saved_image)
            else:
                msg = f"16-bit images can only be saved as TIFF or PNG, not {image_format}"
                raise ValueError(msg)

        elif saved_image.dtype in [np.uint8, np.float32]:
            if s3_client:
                _, encoded_image = cv2.imencode(f".{image_format}", saved_image)
                buffer = io.BytesIO(encoded_image.tobytes())
                # buffer = io.BytesIO()
                # plt.imsave(buffer, saved_image, cmap=cmap, format=image_format)
                # buffer.seek(0)
                upload_file_to_bucket(buffer, img_path, s3_client)
            else:
                cv2.imwrite(img_path, saved_image)
                # plt.imsave(img_path, saved_image, cmap=cmap, format=image_format)

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
        if image.shape[-1] == NUM_CHANNELS_GRAY:
            saved_image = np.squeeze(image, axis=-1)
            cmap = "gray"
        elif len(image.shape) == NUM_DIMS_GRAY:
            cmap = "gray"
            saved_image = image
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
        if is_docker:
            output_path = Path("/app/output/")
        if not output_path:
            output_path = self.output_path
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
        return self._generate_html(self.max_images)

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
        return HTML(self._generate_html(max_images))

    def _generate_html(
        self,
        max_images: int = 12,
        image_number: int | None = None,
    ) -> str:
        """Generate the HTML representation of the object.

        Args:
            max_images (int): The maximum number of images to show. Defaults to 12.
            image_number (int, optional): The image number to show. Defaults to None.

        Returns:
            str: The HTML representation of the object
        """
        html = """
        <style>
        .step-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 5px;
        }
        .metadata {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-left: 20px;
        }
        .image-container img {
            max-width: 200px;
            margin: 10px;
        }
        .toggle-arrow {
            cursor: pointer;
            margin-left: 5px;
        }
        .show-more-button, .hide-button {
            margin-left: 20px;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
        }
        </style>
        <script>
        function toggleMetadata(stepIndex) {
            var x = document.getElementById("metadata-" + stepIndex);
            var arrow = document.getElementById("arrow-" + stepIndex);
            if (x.style.display === "none") {
                x.style.display = "block";
                arrow.innerHTML = "▼";
            } else {
                x.style.display = "none";
                arrow.innerHTML = "►";
            }
        }
        function toggleImage(imageId) {
            var img = document.getElementById(imageId);
            var arrow = document.getElementById("arrow-" + imageId);
            if (img.style.display === "none") {
                img.style.display = "block";
                arrow.innerHTML = "▼";
            } else {
                img.style.display = "none";
                arrow.innerHTML = "►";
            }
        }
        function showMore(stepIndex) {
            var moreImages = document.getElementById("more-images-" + stepIndex);
            var showMoreButton = document.getElementById("show-more-button-" + stepIndex);
            var hideButton = document.getElementById("hide-button-" + stepIndex);
            moreImages.style.display = "flex";
            showMoreButton.style.display = "none";
            hideButton.style.display = "inline-block";
        }
        function hide(stepIndex) {
            var moreImages = document.getElementById("more-images-" + stepIndex);
            var showMoreButton = document.getElementById("show-more-button-" + stepIndex);
            var hideButton = document.getElementById("hide-button-" + stepIndex);
            moreImages.style.display = "none";
            showMoreButton.style.display = "inline-block";
            hideButton.style.display = "none";
        }
        </script>
        """

        for step_index, (step, image_arrays) in enumerate(
            zip(self.steps, self.images, strict=False),
        ):
            html += f"""
                <div class='step-header'>Step: {step} <span id='arrow-{step_index}'
                    class='toggle-arrow'
                    onclick='toggleMetadata({step_index})'>►</span>
                </div>
                """
            html += f"<div id='metadata-{step_index}' class='metadata' style='display:block;'>"
            if image_number is not None:
                images_to_show = [image_arrays[image_number]] if len(image_arrays) > image_number else []
                second_set_images = 0
            else:
                first_set_images = min(max_images, MAX_IMAGES_TO_SHOW)
                second_set_images = max_images - first_set_images if max_images > MAX_IMAGES_TO_SHOW else 0
                images_to_show = image_arrays[:first_set_images]
            html += "<div class='image-container'>"
            if len(images_to_show) == 0:
                html += "<p>No images to show</p>"
                html += "</div>"
            else:
                size = (250, 250) if image_number is None else None

                for image_index, image_array in enumerate(images_to_show):
                    html += self._generate_single_image_html(
                        image_array,
                        step_index,
                        image_index,
                        size,
                    )
                html += "</div>"
                if second_set_images > 0:
                    html += f"""
                        <button id='hide-button-{step_index}' class='hide-button'
                            style='display:block;' onclick='hide({step_index})'>
                            HIDE
                        </button>
                        """
                    html += f"""<div id='more-images-{step_index}' class='image-container'
                        style='display:block;'>
                        """
                    for image_index, image_array in enumerate(
                        image_arrays[12:max_images],
                        start=max_images,
                    ):
                        html += self._generate_single_image_html(
                            image_array,
                            step_index,
                            image_index,
                            size,
                        )
                    html += "</div>"
                    html += f"""
                        <button id='show-more-button-{step_index}' class='show-more-button'
                            onclick='showMore({step_index})'>
                            SHOW MORE
                        </button>
                        """
            html += "</div>"
        return html

    def _generate_single_image_html(
        self,
        image_array: np.ndarray | da.core.Array,
        step_index: int,
        image_index: int,
        size: tuple,
    ) -> str:
        image_id = f"image-{step_index}-{image_index}"
        html = f"""
            <div>
                <p onclick='toggleImage(\"{image_id}\")' style='cursor:pointer;'>
                    Image: {self.filenames[step_index][image_index]}
                    <span id='arrow-{image_id}' class='toggle-arrow'>►</span>
                </p>
            """
        if image_array is not None:
            html += f"""
                <img id='{image_id}'
                    src='{ImagesLayer.numpy_array_to_base64(image_array, size)}'
                    style='display:block;'/></div>
                """
        else:
            html += f"""
                <p id='{image_id}' style='color:red; display:block;'>
                    No image to show
                </p></div>
                """
        return html

    @staticmethod
    def numpy_array_to_base64(
        image_array: np.ndarray | da.core.Array,
        size: tuple = (150, 150),
    ) -> str:
        """Convert a numpy array to a base64 image.

        Args:
            image_array (np.ndarray | da.core.Array): The image array
            size (tuple, optional): _description_. Defaults to (150, 150).

        Returns:
            str: The base64 image
        """
        if isinstance(image_array, da.core.Array):
            image_array = image_array.compute()
        if image_array.shape[-1] == NUM_CHANNELS_GRAY:
            image_array = np.squeeze(image_array, axis=-1)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        if image_array.ndim == NUM_DIMS_GRAY:
            pil_img = Image.fromarray(image_array, mode="L")
        elif image_array.shape[-1] == NUM_CHANNELS_RGBA:
            if image_array[:, :, 3].max() <= 1:
                image_array[:, :, 3] = (image_array[:, :, 3] * 255).astype(np.uint8)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(image_array, mode="RGBA")
        else:
            pil_img = Image.fromarray(image_array, mode="RGB")
        if size:
            pil_img.thumbnail(size)
        buffer = BytesIO()
        img_format = "PNG" if image_array.shape[-1] == NUM_CHANNELS_RGBA else "JPEG"
        pil_img.save(buffer, format=img_format)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{img_format.lower()};base64,{img_str}"
