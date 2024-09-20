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
import dask.array as da

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
        self.filenames = []

    def add_step(
        self,
        step: tuple,
        images: np.ndarray = None,
        catalog: pd.DataFrame = None,
        step_metadata: List[dict] = None,
        update_catalog: bool = False,
    ):
        if update_catalog:
            last_images = self.images[-1]
            print(len(last_images), last_images[0].shape, type(last_images[0]))

            last_filenames = np.array(self.filenames[-1])
            new_filenames = np.isin(last_filenames, catalog["filename"])
            new_images = [image for image, filename in zip(last_images, new_filenames) if filename]
            print(len(new_images), new_images[0].shape, type(new_images[0]))
            self.images.append(new_images)
        else:
            self.images.append(images)
        self.step_metadata.append(step_metadata)
        self.steps.append(step)
        self.filenames.append(catalog["filename"].tolist())

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
        self.filenames = self.filenames[:index]
        return index

    def remove_steps_by_order(self, step_order: int):
        """Remove steps by order

        Args:
            step_order (int): The step order to remove
        """
        self.steps = self.steps[:step_order]
        self.images = self.images[:step_order]
        self.filenames = self.filenames[:step_order]

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
            plt.subplot(int(f"{len(self.images)}1{idx+1}"))
            plt.title(self.steps[idx])
            if images[index].shape[-1] == 1:
                plt.imshow(np.squeeze(images[index], axis=-1), cmap="gray")
            else:
                plt.imshow(images[index])
            # images[index].show(
            #     int(f"{len(self.images)}1{idx+1}"), title=self.steps[idx]
            # )

    def save(self, step: tuple = None, by_order: bool = False, last: bool = False, output_path: str = None, image_format: str = "png"):
        images = self.get_step(step, by_order, last)
        if last:
            step_order = len(self.steps) - 1
        elif by_order:
            step_order = step
        else:
            step_order = self.steps.index(step)
        if not output_path:
            output_path = self.output_path
        for idx, image in enumerate(images):
            img_path = output_path / f"{self.filenames[step_order][idx]}.{image_format}"
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
                cmap = "gray"
            else:
                cmap = None
            plt.imsave(str(img_path), image, cmap=cmap)

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
                html += f"<div><p onclick='toggleImage(\"{image_id}\")' style='cursor:pointer;'>Image: {self.filenames[step_index][image_index]} <span id='arrow-{image_id}' class='toggle-arrow'>►</span></p>"
                html += f"<img id='{image_id}' src='{ImagesLayer.numpy_array_to_base64(image_array)}' style='display:block;' /></div>"
            html += "</div>"
            if second_set_images > 0:
                html += f"<button id='hide-button-{step_index}' class='hide-button' style='display:block;' onclick='hide({step_index})'>HIDE</button>"
                html += f"<div id='more-images-{step_index}' class='image-container' style='display:block;'>"
                for image_index, image_array in enumerate(
                    image_arrays[12:max_images], start=max_images
                ):
                    image_id = f"image-{step_index}-{image_index}"
                    html += f"<div><p onclick='toggleImage(\"{image_id}\")' style='cursor:pointer;'>Image: {self.filenames[step_index][image_index]} <span id='arrow-{image_id}' class='toggle-arrow'>►</span></p>"
                    html += f"<img id='{image_id}' src='{ImagesLayer.numpy_array_to_base64(image_array)}' style='display:block;' /></div>"
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
        # print(image_array.shape)
        # print(image_array.dtype)
        # print(image_array)
        if image_array.shape[-1] == 1:
            image_array = np.squeeze(image_array, axis=-1)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        if image_array.ndim == 2:  # Grayscale image
            pil_img = Image.fromarray(image_array, mode='L')
        else:  # Color image (assume RGB)
            pil_img = Image.fromarray(image_array, mode='RGB')
        pil_img.thumbnail(size)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
