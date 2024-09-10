""" Module to handle images and metadata for each step in the pipeline
"""

from typing import List
from io import BytesIO

import matplotlib.pyplot as plt
import base64
from PIL import Image
from IPython.display import HTML
import pandas as pd
import numpy as np

from paidiverpy.image_layer import ImageLayer


class ImagesLayer:
    """Class to handle images and metadata for each step in the pipeline

    Args:
        output_path (str): Path to save the images. Default is None.
    """

    def __init__(self, output_path=None):
        self.steps = []
        self.step_metadata = []
        self.images = []
        self.max_images = 12
        self.output_path = output_path

    def get_all_filenames(self, step_order: str = None):
        """Get all filenames for a given step order

        Args:
            step_order (int): The step order to get the filenames from. Default is None.

        Returns:
            list: A list of filenames
        """

        return [
            image.get_filename()
            for image in self.images[step_order if step_order else -1]
        ]

    def add_step(
        self,
        step: tuple,
        images: List[ImageLayer] = None,
        catalog: pd.DataFrame = None,
        step_metadata: List[dict] = None,
    ):
        """Add a step to the pipeline

        Args:
            step (tuple): A tuple with the step name and the step order.
            images (List[ImageLayer], optional): A list of ImageLayer objects. Defaults to None.
            catalog (pd.DataFrame, optional): A catalog with the filenames. Defaults to None.
            step_metadata (List[dict], optional): A list of dictionaries with the metadata for each image. Defaults to None.
        """
        if "trim" in step:
            images = self.images[-1]
            images_names = self.get_all_filenames()
            new_images = []
            for image in catalog["filename"]:
                if image in images_names:
                    index = images_names.index(image)
                    new_images.append(images[index])
            self.images.append(new_images)
        else:
            self.images.append(images)
        self.step_metadata.append(step_metadata)
        self.steps.append(step)

    def remove_steps_by_name(self, step: tuple):
        """Remove steps by name

        Args:
            step (tuple): The step to remove

        Returns:
            int: The index of the removed step
        """
        index = self.steps.index(step)
        self.steps = self.steps[:index]
        self.images = self.images[:index]
        return index

    def remove_steps_by_order(self, step_order: int):
        """Remove steps by order

        Args:
            step_order (int): The step order to remove
        """
        self.steps = self.steps[:step_order]
        self.images = self.images[:step_order]

    def get_last_step_order(self):
        """Get the last step order

        Returns:
            int: The last step order
        """
        return len(self.steps) - 1

    def get_step(self, step: tuple = None, by_order: bool = False, last: bool = False):
        """Get a step by name or order

        Args:
            step (tuple, optional): The step to get. Defaults to None.
            by_order (bool, optional): If True, get the step by order. Defaults to False.
            last (bool, optional): If True, get the last step. Defaults to False.

        Returns:
            List[Image]: A list of Image objects
        """
        if last:
            return self.images[-1]
        if by_order:
            index = step
        else:
            index = self.steps.index(step)
        return self.images[index]

    def show(self, index: int = 10):
        """Show the images in the pipeline

        Args:
            index (int, optional): The index of the image to show. Defaults to 10.
        """
        plt.figure(figsize=(20, 20))

        for idx, images in enumerate(self.images):
            images[index].show(
                int(f"{len(self.images)}1{idx+1}"), title=self.steps[idx]
            )

    def save(
        self, output_path: str = None, filename: str = None, image_format: str = "png"
    ):
        """Save the images in the pipeline

        Args:
            output_path (str, optional): The path to save the images. Defaults to None.
            filename (str, optional): The filename to save the images. Defaults to None.
            image_format (str, optional): The image format to save the images. Defaults to "png".
        """
        pass
        # if isinstance(output_path, str):
        #     output_path = Path(output_path)
        # if not filename:
        #     filename = f"output_{self.get_filename().split('.')[0]}_{self.step_order}"
        # img_path = output_path / f"{filename}.{image_format}"
        # cv2.imwrite(str(img_path), self.image)
        # self.logger.info("Image saved to %s", img_path)

    def __repr__(self) -> str:
        """Return the string representation of the object

        Returns:
            str: The string representation of the object
        """
        repr_str = ""
        for step, image_paths in zip(self.steps, self.images):
            repr_str += f"Step: {step}\n"
            for image_path in image_paths:
                repr_str += f"Image: {image_path}\n"
        return repr_str

    def _repr_html_(self) -> str:
        """Return the HTML representation of the object

        Returns:
            str: The HTML representation of the object
        """
        return self._generate_html(self.max_images)

    def __call__(self, max_images: int = None) -> HTML:
        """Call the object

        Args:
            max_images (int, optional): The maximum number of images to show. Defaults to None.

        Returns:
            HTML: The HTML representation of the object
        """
        if not max_images:
            max_images = self.max_images
        return HTML(self._generate_html(max_images))

    def _generate_html(self, max_images: int) -> str:
        """Generate the HTML representation of the object

        Args:
            max_images (int): The maximum number of images to show

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

        for step_index, (step, image_arrays) in enumerate(zip(self.steps, self.images)):
            html += f"<div class='step-header'>Step: {step} <span id='arrow-{step_index}' class='toggle-arrow' onclick='toggleMetadata({step_index})'>►</span></div>"
            html += f"<div id='metadata-{step_index}' class='metadata' style='display:block;'>"

            first_set_images = 12 if max_images > 12 else max_images
            second_set_images = max_images - first_set_images if max_images > 12 else 0
            images_to_show = image_arrays[:first_set_images]
            html += "<div class='image-container'>"
            for image_index, image_array in enumerate(images_to_show):
                image_id = f"image-{step_index}-{image_index}"
                html += f"<div><p onclick='toggleImage(\"{image_id}\")' style='cursor:pointer;'>Image: {image_array.image_metadata['filename']} <span id='arrow-{image_id}' class='toggle-arrow'>►</span></p>"
                html += f"<img id='{image_id}' src='{ImagesLayer.numpy_array_to_base64(image_array.image)}' style='display:block;' /></div>"
            html += "</div>"
            if second_set_images > 0:
                html += f"<button id='hide-button-{step_index}' class='hide-button' style='display:block;' onclick='hide({step_index})'>HIDE</button>"
                html += f"<div id='more-images-{step_index}' class='image-container' style='display:block;'>"
                for image_index, image_array in enumerate(
                    image_arrays[12:max_images], start=max_images
                ):
                    image_id = f"image-{step_index}-{image_index}"
                    html += f"<div><p onclick='toggleImage(\"{image_id}\")' style='cursor:pointer;'>Image: {image_array.image_metadata['filename']} <span id='arrow-{image_id}' class='toggle-arrow'>►</span></p>"
                    html += f"<img id='{image_id}' src='{ImagesLayer.numpy_array_to_base64(image_array.image)}' style='display:block;' /></div>"
                html += "</div>"
                html += f"<button id='show-more-button-{step_index}' class='show-more-button' onclick='showMore({step_index})'>SHOW MORE</button>"

            html += "</div>"

        return html

    @staticmethod
    def numpy_array_to_base64(image_array: np.ndarray, size: tuple = (150, 150)) -> str:
        """Convert a numpy array to a base64 image

        Args:
            image_array (np.ndarray): The image array
            size (tuple, optional): _description_. Defaults to (150, 150).

        Returns:
            str: The base64 image
        """
        pil_img = Image.fromarray(image_array)
        if pil_img.mode == "F":
            pil_img = pil_img.convert("L")
        pil_img.thumbnail(size)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
