"""HTML output utilities for paidiverpy."""

import base64
from functools import lru_cache
from html import escape
from importlib.resources import files
from io import BytesIO
from typing import TYPE_CHECKING
import dask.array as da
import numpy as np
from IPython.display import HTML
from PIL import Image
from paidiverpy.utils.data import NUM_CHANNELS_GREY
from paidiverpy.utils.data import NUM_CHANNELS_RGBA
from paidiverpy.utils.data import NUM_DIMENSIONS
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY

if TYPE_CHECKING:
    from paidiverpy import Paidiverpy
    from paidiverpy.config.configuration import Configuration
    from paidiverpy.images_layer import ImagesLayer
    from paidiverpy.metadata_parser import MetadataParser

MAX_IMAGES_TO_SHOW = 12

STATIC_FILES = (
    ("paidiverpy.static.html", "icons-svg-inline.html"),
    ("paidiverpy.static.css", "style.css"),
    ("paidiverpy.static.js", "script.js"),
)

EXTERNAL_CSS = ["https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"]

EXTERNAL_JS = ("https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js", "hljs.highlightAll();")


@lru_cache(None)
def _load_static_files() -> list[str]:
    """Lazily load the resource files into memory the first time they are needed.

    Returns:
        list[str]: List of strings containing the contents of the static files
    """
    return [files(package).joinpath(resource).read_text(encoding="utf-8") for package, resource in STATIC_FILES]


def _obj_repr(obj: object, body: list[str], html: bool = False) -> str | HTML:
    """Return HTML repr of an xarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.

    Args:
        obj (object): The object to represent
        body (list[str]): The HTML string
        html (bool): If True, the output will be in HTML format. Defaults to False.

    Returns:
        str: The HTML representation of the object
    """
    icons_svg, css_style, script = _load_static_files()
    external_css = "\n".join(f'<link rel="stylesheet" href="{url}">' for url in EXTERNAL_CSS)
    external_js = ""
    for url in EXTERNAL_JS:
        if url.endswith(".js"):
            external_js += f'<script src="{url}"></script>'
        else:
            external_js += f"<script>{url}</script>"
    html_str = (
        "<div>"
        f"{icons_svg}<style>{css_style}</style><script>{script}</script>"
        f"{external_css}{external_js}"
        f"<pre class='ppy-text-repr-fallback'>{escape(repr(obj))}</pre>"
        f"<div>{body}</div>"
        "</div>"
    )

    return html_str if not html else HTML(html_str)


def _icon(icon_name: str) -> str:
    """Return HTML representation of an icon.

    Args:
        icon_name (str): The name of the icon to represent.

    Returns:
        str: The HTML representation of the icon.
    """
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return f"<svg class='icon ppy-{icon_name}'><use xlink:href='#{icon_name}'></use></svg>"


def metadata_repr(metadata: "MetadataParser") -> str:
    """Represents metadata as an HTML string.

    Args:
        metadata (MetadataParser): The metadata object to represent.

    Returns:
        str: String representation of the metadata.
    """
    message = "This is a instance of 'MetadataParser'<br><br>"
    if metadata.dataset_metadata:
        message += "<b>Dataset Metadata:</b><br>"
        message += _json_to_html(metadata.dataset_metadata)
    message += "<b>Images Metadata:</b><br>"
    body = message + metadata.metadata._repr_html_()
    return _obj_repr(metadata, body)


def pipeline_repr(pipeline: "Paidiverpy") -> str:
    """Generate HTML representation of the pipeline.

    Args:
        pipeline (Paidiverpy): The pipeline instance to represent.

    Returns:
        str: The HTML representation of the pipeline.
    """
    steps_html = ""
    parameters_html = ""

    random_id = "".join([chr(np.random.default_rng().integers(97, 122)) for _ in range(3)])

    for i, step in enumerate(pipeline.config.steps):
        if i % 4 == 0 and i > 0:
            steps_html += '<div class="ppy-pipeline-clear-fix"></div>'
        steps_html += f"""
            <div id="ppy-pipeline-{random_id}-step-{i}" title="Click to see more information"
                class="ppy-pipeline ppy-pipeline-{random_id}" onclick="showParameters('step-{i}', '{random_id}')">
                <h2 class="ppy-h2">{step.name.capitalize()}</h2>
                <h3 class="ppy-h3">Type: {step.step_name.capitalize()}</h2>
            </div>
        """
        if i < len(pipeline.config.steps) - 1:
            steps_html += f"<div class='ppy-pipeline-arrow'>{_icon('arrow-right')}</div>"
        parameters_html += f"""
            <div id="ppy-pipeline-parameters-{random_id}-step-{i}"
                class="ppy-pipeline-parameters ppy-pipeline-parameters-{random_id}"
                style="display: none;">
                {_json_to_html(step.to_dict())}
            </div>
        """

    general_html = f"""
    <div id="ppy-pipeline-{random_id}-general" title="Click to see more information"
        class="ppy-pipeline ppy-pipeline-{random_id}"
        onclick="showParameters('general', '{random_id}')">
        <h2 class="ppy-h2">{pipeline.config.general.name.capitalize()}</h2>
        <h3 class="ppy-h3">Type: {pipeline.config.general.step_name.capitalize()}</h2>
    </div>
    """

    parameters_html += f"""
        <div id="ppy-pipeline-parameters-{random_id}-general" class="ppy-pipeline-parameters ppy-pipeline-parameters-{random_id}"
            style="display: none;">
            {_json_to_html(pipeline.config.general.to_dict())}
        </div>
    """

    part_text = ""
    if len(pipeline.steps) > 1:
        part_text = f'<div class="ppy-pipeline-arrow">{_icon("arrow-right")}</div>{steps_html}'

    body = f"""
    <div class="ppy-pipeline-wrap">
        {general_html}{part_text}
    </div>
    <div id="ppy-pipeline-parameters" class="ppy-pipeline-parameters-all">{parameters_html}</div>
    """
    return _obj_repr(pipeline, body)


def config_repr(config: "Configuration") -> str:
    """Generate HTML representation of the config.

    Args:
        config (Configuration): The configuration instance to represent.

    Returns:
        str: The HTML representation of the config.
    """
    config_html_str = _json_to_html(config.to_dict())
    return _obj_repr(config, config_html_str)


def images_repr(
    images: "ImagesLayer",
    max_images: int = 12,
    image_number: int | None = None,
    html: bool = False,
) -> str:
    """Generate the HTML representation of the object.

    Args:
        images (ImagesLayer): The ImagesLayer object to represent.
        max_images (int): The maximum number of images to show. Defaults to 12.
        image_number (int, optional): The image number to show. Defaults to None.
        html (bool): If True, the output will be in HTML format. Defaults to False.

    Returns:
        str: The HTML representation of the object
    """
    body = ""
    # generate a ramdon 3 characters string
    random_id = "".join([chr(np.random.default_rng().integers(97, 122)) for _ in range(3)])

    for step_index, (step, image_arrays) in enumerate(
        zip(images.steps, images.images, strict=False),
    ):
        body += f"""
            <div class='ppy-h2 ppy-images-step-header' onclick='toggleMetadata({step_index}, "{random_id}")'>
                Step {step_index}: {step}
                <span id='ppy-images-arrow-{random_id}-{step_index}'
                    class='ppy-images-toggle-arrow ppy-font-color-brown'>▼
                </span>
            </div>
            """
        body += f"<div id='ppy-images-metadata-{random_id}-{step_index}' class='ppy-images-metadata' style='display:block;'>"
        if image_number is not None:
            images_to_show = [image_arrays[image_number]] if len(image_arrays) > image_number else []
        else:
            first_set_images = min(max_images, MAX_IMAGES_TO_SHOW)
            images_to_show = image_arrays[:first_set_images]
        body += "<div class='ppy-images'>"
        if len(images_to_show) == 0:
            body += "<p class='ppy-p-error'>No images to show</p>"
            body += "</div>"
        else:
            size = (250, 250) if image_number is None else None

            for image_index, image_array in enumerate(images_to_show):
                body += generate_single_image_html(image_array, images.filenames, step_index, image_index, size, random_id)
            body += "</div>"
        body += "</div>"
    return _obj_repr(images, body, html=html)


def generate_single_image_html(
    image_array: np.ndarray | da.core.Array,
    filenames: list[str],
    step_index: int,
    image_index: int,
    size: tuple,
    random_id: str,
) -> str:
    """Generate HTML for a single image.

    Args:
        image_array (np.ndarray | da.core.Array): The image array
        filenames (list[str]): The filenames of the images
        step_index (int): The index of the step
        image_index (int): The index of the image
        size (tuple): The size of the image
        random_id (str): The random id for the image

    Returns:
        str: The HTML for the image
    """
    image_id = f"image-{random_id}-{step_index}-{image_index}"
    html = f"""
        <div>
            <p onclick='toggleImage("{image_id}", "{random_id}")' class="ppy-images-image-p" >
                Image: {filenames[step_index][image_index]}
                <span id='ppy-images-arrow-{random_id}-{image_id}' class='ppy-images-toggle-arrow ppy-font-color-brown'>▼</span>
            </p>
        """
    if image_array is not None:
        html += f"""
            <img id='{image_id}'
                src='{numpy_array_to_base64(image_array, size)}'
                class='ppy-images-img'/></div>
            """
    else:
        html += f"""
            <p id='{image_id}' class='ppy-p ppy-p-error'>
                No image to show
            </p></div>
            """
    return html


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
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    if image_array.shape[-1] == NUM_CHANNELS_GREY:
        image_array = np.squeeze(image_array, axis=-1)
    if image_array.ndim == NUM_DIMENSIONS_GREY:
        pil_img = Image.fromarray(image_array, mode="L")
    elif image_array.shape[-1] == NUM_CHANNELS_RGBA:
        if image_array[:, :, 3].max() <= 1:
            image_array[:, :, 3] = (image_array[:, :, 3] * 255).astype(np.uint8)
        # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(image_array, mode="RGBA")
    else:
        pil_img = Image.fromarray(image_array, mode="RGB")
    if size:
        pil_img.thumbnail(size)
    buffer = BytesIO()
    img_format = "PNG" if image_array.ndim == NUM_DIMENSIONS and image_array.shape[-1] == NUM_CHANNELS_RGBA else "JPEG"
    pil_img.save(buffer, format=img_format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{img_format.lower()};base64,{img_str}"


def _json_to_html(data: dict | list) -> str:
    """Convert JSON data to HTML.

    Args:
        data (dict | list): The JSON data to convert.

    Returns:
        str: The HTML representation of the JSON data.
    """

    def style(value: object) -> str:
        """Style the value based on its type.

        Args:
            value (object): The value to style.

        Returns:
            str: The styled value.
        """
        if isinstance(value, str):
            return f'<span class="ppy-json-string">"{escape(value)}"</span>'
        if isinstance(value, int | float):
            return f'<span class="ppy-json-number">{value}</span>'
        if isinstance(value, bool):
            return f'<span class="ppy-json-bool">{str(value).lower()}</span>'
        if value is None:
            return '<span class="ppy-json-null">null</span>'
        return str(value)

    def render(obj: object, indent: int = 0) -> str:
        """Render the object as HTML.

        Args:
            obj (object): The object to render.
            indent (int): The current indentation level.

        Returns:
            str: The rendered HTML.
        """
        spacing = " " * (indent * 4)
        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                key_html = f'<span class="ppy-json-key">"{escape(str(k))}"</span>'
                val_html = render(v, indent + 1)
                items.append(f"{spacing}    {key_html}: {val_html}")
            return "{\n" + ",\n".join(items) + f"\n{spacing}}}"
        if isinstance(obj, list):
            items = [render(i, indent + 1) for i in obj]
            return "[\n" + ",\n".join(f"{spacing}    {i}" for i in items) + f"\n{spacing}]"
        return style(obj)

    return f'<pre class="ppy-json-block">{render(data)}</pre>'
