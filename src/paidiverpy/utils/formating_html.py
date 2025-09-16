"""HTML output utilities for paidiverpy."""

import base64
from functools import lru_cache
from html import escape
from importlib.resources import files
from io import BytesIO
from typing import TYPE_CHECKING
from typing import Any
import dask.array as da
import numpy as np
from IPython.display import HTML
from PIL import Image
from shapely import Polygon
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
    ("paidiverpy.utils.static.html", "icons-svg-inline.html"),
    ("paidiverpy.utils.static.css", "style.css"),
    ("paidiverpy.utils.static.js", "script.js"),
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


def _obj_repr(obj: object, body: str, html: bool = False) -> str | HTML:
    """Return HTML repr of an xarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.

    Args:
        obj (object): The object to represent
        body (str): The HTML string
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
    # if isinstance(metadata.metadata, dd.DataFrame):
    #     message += "Metadata is dask dataframe. To see it, please run 'metadata.compute()' through the MetadataParser layer.<br>"
    #     message += "From your processing instance, you can also run the command 'instance_class.metadata.compute()'<br>"
    #     body = message

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
                <h4 class="ppy-h4">Parameters:</h4>
                {_yaml_to_html(step.to_dict())}
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
            <h4 class="ppy-h4">Parameters:</h4>
            {_yaml_to_html(pipeline.config.general.to_dict())}
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
    config_html_str = _yaml_to_html(config.to_dict())
    return _obj_repr(config, config_html_str)


def images_repr(
    images: "ImagesLayer",
    max_images: int = 12,
    image_number: int | None = None,
    html: bool = False,
) -> str | HTML:
    """Generate the HTML representation of the object.

    Args:
        images (ImagesLayer): The ImagesLayer object to represent.
        max_images (int): The maximum number of images to show. Defaults to 12.
        image_number (int, optional): The image number to show. Defaults to None.
        html (bool): If True, the output will be in HTML format. Defaults to False.

    Returns:
        str | HTML: The HTML representation of the object.
    """
    body = ""
    # generate a ramdon 3 characters string
    random_id = "".join([chr(np.random.default_rng().integers(97, 122)) for _ in range(3)])

    for step_index, step in enumerate(images.steps):
        body += f"""
            <div class='ppy-h2 ppy-images-step-header' onclick='toggleMetadata({step_index}, "{random_id}")'>
                Step {step_index}: {step}
                <span id='ppy-images-arrow-{random_id}-{step_index}'
                    class='ppy-images-toggle-arrow ppy-font-color-brown'>▼
                </span>
            </div>
            """
        body += f"<div id='ppy-images-metadata-{random_id}-{step_index}' class='ppy-images-metadata' style='display:block;'>"

        dataset = images.get_step(step=step_index, last=False)
        if not dataset:
            body += "<div class='ppy-images'><p class='ppy-p-error'>Images for this step are not available</p></div>"
        else:
            keep_filenames = np.where((images.images.flag.values == 0) | (images.images.flag.values > step_index))[0]
            dataset = dataset.isel(filename=keep_filenames)
            filenames = dataset["filename"].data

            if image_number is not None:
                dataset = dataset.isel(filename=[image_number]) if len(filenames) > image_number else dataset.isel(filename=[])
            else:
                first_set_images = min(max_images, len(filenames))
                dataset = dataset.isel(filename=range(first_set_images))

            size = (250, 250) if image_number is None else None
            html_snippets = [
                generate_single_image_html(
                    dataset["images"].isel(filename=image_index).to_numpy(),
                    dataset["original_height"].isel(filename=image_index).item(),
                    dataset["original_width"].isel(filename=image_index).item(),
                    dataset["filename"].isel(filename=image_index).item(),
                    step_index,
                    image_index,
                    size,
                    random_id,
                )
                for image_index in range(dataset.sizes["filename"])
            ]
            body += "<div class='ppy-images'>"
            for snippet in html_snippets:
                body += snippet
            body += "</div>"
        body += "</div>"
    return _obj_repr(images, body, html=html)


def generate_single_image_html(
    image_array: np.ndarray[Any, Any] | da.core.Array,
    height: int,
    width: int,
    filename: str,
    step_index: int,
    image_index: int,
    size: tuple[int, int] | None,
    random_id: str,
) -> str:
    """Generate HTML for a single image.

    Args:
        image_array (np.ndarray | da.core.Array): The image array
        height (int): The height of the image
        width (int): The width of the image
        filename (str): The filename of the image
        step_index (int): The index of the step
        image_index (int): The index of the image
        size (tuple): The size of the image
        random_id (str): The random id for the image

    Returns:
        str: The HTML for the image
    """
    image_array = image_array[:height, :width, :]
    image_id = f"image-{random_id}-{step_index}-{image_index}"
    html = f"""
        <div>
            <p onclick='toggleImage("{image_id}", "{random_id}")' class="ppy-images-image-p" >
                Image: {filename}
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
    image_array: np.ndarray[Any, Any] | da.core.Array,
    size: tuple[int, int] | None,
) -> str:
    """Convert a numpy array to a base64 image.

    Args:
        image_array (np.ndarray | da.core.Array): The image array
        size (tuple): The size of the image

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


def _json_to_html(data: dict[str, Any] | list[Any]) -> str:
    """Convert JSON data to HTML.

    Args:
        data (dict | list): The JSON data to convert.

    Returns:
        str: The HTML representation of the JSON data.
    """

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
            items: list[str] = []
            for key, value in obj.items():
                key_html = f'<span class="ppy-json-key">"{escape(str(key))}"</span>'
                val_html = render(value, indent + 1)
                items.append(f"{spacing}    {key_html}: {val_html}")
            return "{\n" + ",\n".join(items) + f"\n{spacing}}}"
        if isinstance(obj, list):
            items = [render(item, indent + 1) for item in obj]
            return "[\n" + ",\n".join(f"{spacing}    {item}" for item in items) + f"\n{spacing}]"
        return style_json_yaml(obj)

    return f'<pre class="ppy-json-block">{render(data)}</pre>'


def _yaml_to_html(data: dict[str, Any] | list[Any]) -> str:
    """Convert dict or list data to styled YAML in HTML.

    Args:
        data (dict | list): The data to convert.

    Returns:
        str: The HTML representation of the YAML data.
    """

    def render(obj: object, indent: int = 0) -> str:
        """Render dicts/lists into YAML format."""
        spacing = " " * (indent * 2)
        if isinstance(obj, dict):
            lines: list[str] = []
            for key, value in obj.items():
                key_html = f'<span class="ppy-json-key">{escape(str(key))}</span>:'
                if isinstance(value, dict | list):
                    val_html = render(value, indent + 1)
                    lines.append(f"{spacing}{key_html}\n{val_html}")
                else:
                    val_html = style_json_yaml(value)
                    lines.append(f"{spacing}{key_html} {val_html}")
            return "\n".join(lines)

        if isinstance(obj, list):
            lines = []
            for item in obj:
                if isinstance(item, dict | list):
                    val_html = render(item, indent + 1)
                    lines.append(f"{spacing}- {val_html.strip()}")
                else:
                    val_html = style_json_yaml(item)
                    lines.append(f"{spacing}- {val_html.strip()}")
            return "\n".join(lines)

        return f"{spacing}{style_json_yaml(obj)}"

    return f'<pre class="ppy-json-block">{render(data)}</pre>'


def style_json_yaml(value: Any) -> str:  # noqa: ANN401
    """Style the value based on its type.

    Args:
        value (object): The value to style.

    Returns:
        str: The styled value.
    """
    import pandas as pd

    if isinstance(value, str):
        output = f'<span class="ppy-json-string">"{escape(value)}"</span>'
    elif isinstance(value, int | float):
        output = f'<span class="ppy-json-number">{value}</span>'
    elif isinstance(value, bool):
        output = f'<span class="ppy-json-bool">{str(value).lower()}</span>'
    elif value is None:
        output = '<span class="ppy-json-null">null</span>'
    elif isinstance(value, pd.Series):
        values = value.tolist()
        styles = [style_json_yaml(item) for item in values]
        output = "[" + ", ".join(styles) + "]"
    elif isinstance(value, Polygon):
        output = f'<span class="ppy-json-string">"{escape(str(value))}"</span>'
    else:
        output = str(value)
    return output
