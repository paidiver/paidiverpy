"""Open Layer utils module."""

import logging
from io import BytesIO
from pathlib import Path
import cv2
import dask
import dask.array as da
import numpy as np
import rawpy
from dask import delayed
from PIL import Image
from PIL.ExifTags import TAGS
from paidiverpy.utils.data import EIGHT_BITS
from paidiverpy.utils.data import NUM_CHANNELS_RGB
from paidiverpy.utils.data import NUM_CHANNELS_RGBA
from paidiverpy.utils.data import NUM_DIMENSIONS
from paidiverpy.utils.data import NUM_DIMENSIONS_GREY
from paidiverpy.utils.data import SIXTEEN_BITS
from paidiverpy.utils.object_store import get_file_from_bucket

SUPPORTED_OPENCV_IMAGE_TYPES = [
    "bmp", "dib", # Windows bitmaps
    "jpg", "jpeg", "jpe", # JPEG files
    "jp2", # JPEG 2000 files
    "png", # Portable Network Graphics
    "webP", # WebP
    "pbm", "pgm", "ppm", "pxm", "pnm", # Portable image format
    "sr", "ras", # Sun rasters
    "tiff", "tif", # TIFF files
    "exr", # OpenEXR Image files
    "hdr", "pic", # Radiance HDR
    "", # non specified
]

SUPPORTED_PIL_IMAGE_TYPES = [
    "bmp", "dib", # Windows bitmaps
    "jpg", "jpeg", # JPEG files
    "jp2", # JPEG 2000 files
    "png", # Portable Network Graphics
    "ppm", "pgm", "pbm", # Portable image format
    "tiff", "tif", # TIFF files
    "webp", # WebP
    "", # non specified
]

SUPPORTED_RAWPY_IMAGE_TYPES = [
    "crw", # Canon RAW
    "cr2", # Canon RAW
    "cr3", # Canon RAW
    "dng", # Adobe Digital Negative
    "nef", # Nikon RAW
    "nrw", # Nikon RAW
    "orf", # Olympus RAW
    "rw2", # Panasonic RAW
    "raf", # Fuji RAW
]

def open_image_remote(img_path: str,
                      image_type: str | None,
                      image_open_args: dict | None = None,
                      **kwargs: dict) -> tuple[np.ndarray | dask.array.core.Array, dict]:
    """Open an image file.

    Args:
        img_path (str): The path to the image file
        image_type (str | None): The image type
        image_open_args (dict | None): The image open arguments
        **kwargs (dict): Additional keyword arguments. The following are supported:
            - storage_options (dict): The storage options for reading metadata file.
            - parallel (bool): Whether to use Dask for parallel processing.

    Raises:
        ValueError: Failed to open the image

    Returns:
        tuple[np.ndarray | dask.array.core.Array, dict]: The image data and the EXIF data
    """
    exif = {}
    try:
        img_bytes = get_file_from_bucket(img_path, kwargs.get("storage_options"))
        if image_type in SUPPORTED_OPENCV_IMAGE_TYPES:
            img_array = np.frombuffer(img_bytes, image_open_args.get("dtype", np.uint8))
            img = cv2.imdecode(img_array, image_open_args.get("flags", cv2.IMREAD_UNCHANGED))
            exif = extract_exif_single(BytesIO(img_bytes), image_type=image_type, image_name=img_path.split("/")[-1])
        else:
            img = load_raw_image(BytesIO(img_bytes), image_type=image_type, image_open_args=image_open_args, remote=True)
    except (FileNotFoundError, OSError, TypeError) as e:
        img = None
        logging.warning("Failed to open %s: %s", img_path, e)

    img = correct_image_dims_and_format(img, kwargs.get("parallel"))

    return img, exif


def open_image_local(img_path: str,
                     image_type: str | None,
                     image_open_args: dict | None = None,
                     **kwargs: dict) -> tuple[np.ndarray | dask.array.core.Array, dict]:
    """Open an image file.

    Args:
        img_path (str): The path to the image file
        image_type (str | None): The image type
        image_open_args (dict | None): The image open arguments
        **kwargs (dict): Additional keyword arguments. The following are supported:
            - parallel (bool): Whether to use Dask for parallel processing.

    Raises:
        ValueError: Failed to open the image

    Returns:
        tuple[np.ndarray | dask.array.core.Array, dict]: The image data and the EXIF data
    """
    exif = extract_exif_single(img_path=img_path, image_type=image_type)
    if image_type in SUPPORTED_OPENCV_IMAGE_TYPES:
        img = cv2.imread(str(img_path), image_open_args.get("flags", cv2.IMREAD_UNCHANGED))
    else:
        img = load_raw_image(img_path, image_type=image_type, image_open_args=image_open_args)
    img = correct_image_dims_and_format(img, kwargs.get("parallel"))
    return img, exif

def correct_image_dims_and_format(img: np.ndarray | dask.array.core.Array, parallel: bool) -> np.ndarray | dask.array.core.Array:
    """Correct the image dimensions and format.

    Args:
        img (np.ndarray | dask.array.core.Array): The image data
        parallel (bool): Whether to use Dask for parallel processing

    Returns:
        np.ndarray | dask.array.core.Array: The corrected image data
    """
    if img is None:
        return img
    if img.ndim == NUM_DIMENSIONS_GREY:
        img = np.expand_dims(img, axis=-1)
    elif img.ndim == NUM_DIMENSIONS and img.shape[2] == NUM_CHANNELS_RGBA:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    elif img.ndim == NUM_DIMENSIONS and img.shape[2] == NUM_CHANNELS_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if parallel:
        img = da.from_array(img, chunks=img.shape)
    return img


def load_raw_image(img_path: str, image_type: str | None, image_open_args: dict | None, remote: bool = False) -> np.ndarray | dask.array.core.Array:
    """Load a raw image file.

    Args:
        img_path (str): The path to the image file or a BytesIO object
        image_type (str | None): The image type
        image_open_args (dict | None): The image open arguments
        remote (bool): Whether the image is remote or local. Defaults to False.

    Raises:
        ValueError: Failed to open the image

    Returns:
        np.ndarray: The loaded image data
    """
    img = None
    if image_type in SUPPORTED_RAWPY_IMAGE_TYPES:
        try:
            img_bytes = img_path if remote else str(img_path)
            with rawpy.imread(img_bytes) as raw:
                img = raw.postprocess(**image_open_args)
        except rawpy.LibRawFileUnsupportedError as e:
            logging.warning("Failed to open %s using rawpy: %s. Trying using raw loader", img_path, e)
    if img is None:
        try:
            width = image_open_args.get("width", 2448)
            height = image_open_args.get("height", 2048)
            bit_depth = image_open_args.get("bit_depth", 8)
            bayer_pattern = image_open_args.get("bayer_pattern")
            image_format = image_open_args.get("image_format", "mono")
            endianness = image_open_args.get("endianness")
            file_header_size = image_open_args.get("file_header_size", 0)
            channels = image_open_args.get("channels", 1)
            img_bytes = img_path
            if not remote:
                with Path(img_path).open("rb") as file:
                    img_bytes = file.read()
            img_bytes.seek(file_header_size)
            raw_data = img_bytes.read()

            if bit_depth == EIGHT_BITS:
                dtype = np.uint8
            elif bit_depth <= SIXTEEN_BITS:
                dtype = np.dtype("<u2") if endianness == "little" else np.dtype(">u2")
            else:
                msg = "Failed to load the image. Unsupported bit depth"
                raise ValueError(msg)

            img = np.frombuffer(raw_data, dtype=dtype)
            img = img.reshape((height, width, channels)) if channels > 1 else img.reshape((height, width))
            if image_format.lower() == "bayer":
                code = {
                    "BG": cv2.COLOR_BayerBG2RGB,
                    "GB": cv2.COLOR_BayerGB2RGB,
                    "RG": cv2.COLOR_BayerRG2RGB,
                    "GR": cv2.COLOR_BayerGR2RGB
                }[bayer_pattern]
                img = cv2.cvtColor(img, code)
        except (FileNotFoundError, OSError, TypeError) as e:
            logging.warning("Failed to open %s: %s", img_path, e)
    return img


def extract_exif_single(img_path: str, image_type: str, image_name: str | None = None) -> dict:
    """Extract EXIF data from a single image file.

    Args:
        img_path (str): The path to the image file.
        image_type (str): The image type.
        image_name (str, optional): The name of the image file. Defaults to None.

    Returns:
        dict: The EXIF data.
    """
    exif = {}
    if image_type and image_type not in SUPPORTED_PIL_IMAGE_TYPES:
        logging.warning("Image type %s not supported for EXIF extraction", image_type)
        return exif
    try:
        img_pil = Image.open(img_path)
        exif_data = img_pil.getexif()
        if exif_data is not None:
            if image_name:
                exif["image-filename"] = image_name
            else:
                exif["image-filename"] = img_path.name
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif[tag_name] = value
    except FileNotFoundError as e:
        logging.warning("Failed to open %s: %s", img_path, e)
    except OSError as e:
        logging.warning("Failed to open %s: %s", img_path, e)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to extract EXIF data from %s: %s", img_path, e)
    return exif
