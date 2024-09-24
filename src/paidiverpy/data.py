""" Helper functions to download and load datasets. """

import zipfile
import hashlib
import json
from pathlib import Path
import requests

from utils import initialise_logging

logger = initialise_logging(verbose=2)

# Define a base directory for caching
CACHE_DIR = Path.home() / ".paidiverpy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENCE_FILE = CACHE_DIR / "datasets.json"

DATASET_URLS = {
    "pelagic": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/mojtaba_images.zip",
        "metadata_type": "CSV",
        "image_type": "BMP",
    },
    "benthic_csv": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_csv.zip",
        "metadata_type": "CSV",
        "image_type": "JPG",
        "append_data_to_metadata": True
    },
    "benthic_ifdo": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_ifdo.zip",
        "metadata_type": "IFDO",
        "image_type": "RAW",
    }
}


def load_persistent_paths() -> dict:
    """Load the persistent paths from the cache directory.

    Returns:
        dict: The persistent paths.
    """
    if PERSISTENCE_FILE.exists():
        with open(PERSISTENCE_FILE, "r", encoding="UTF-8") as f:
            return json.load(f)
    return {}


def save_persistent_paths(paths: dict) -> None:
    """Save the persistent paths to the cache directory.

    Args:
        paths (dict): The paths to save.
    """

    with open(PERSISTENCE_FILE, "w", encoding="UTF-8") as f:
        json.dump(paths, f)


def download_file(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """
    Download the file from the given URL and cache it locally to avoid redundant downloads.

    Args:
        url (str): The URL to download the file from.
        cache_dir (Path): The directory to store the downloaded file.

    Returns:
        Path: The path to the downloaded file.
    """
    file_hash = hashlib.md5(url.encode()).hexdigest()
    zip_path = cache_dir / f"{file_hash}.zip"

    if not zip_path.exists():
        logger.info("Downloading %s...", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(response.content)
        logger.info("Downloaded and cached at %s", zip_path)

    return zip_path


def unzip_file(zip_path: Path, extract_dir: Path = CACHE_DIR) -> None:
    """
    Unzip the file to the specified directory.

    Args:
        zip_path (Path): The path to the zip file.
        extract_dir (Path): The directory to extract the contents to.
    """
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info("Extracted files to %s", extract_dir)
    else:
        logger.info("Using cached extraction at %s", extract_dir)


def load(dataset_name: str) -> dict:
    """
    Download, unzip, and load the specified dataset.

    Args:
        dataset_name (str): The name of the dataset (for example, 'sample_image').

    Returns:
        dict: A dictionary containing the input path, metadata path, metadata type, and image type.
    """
    paths = load_persistent_paths()

    if dataset_name in paths and Path(paths[dataset_name]).exists():
        logger.info(
            "Dataset '%s' is already cached at %s", dataset_name, paths[dataset_name]
        )
        return paths[dataset_name]

    extract_dir = CACHE_DIR / dataset_name
    dataset_information = DATASET_URLS.get(dataset_name)
    if dataset_information is None:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    url = dataset_information["url"]
    zip_path = download_file(url)
    unzip_file(zip_path, extract_dir)

    if zip_path.exists():
        zip_path.unlink()

    logger.info("Dataset '%s' is available at: %s", dataset_name, extract_dir)
    information = {
        "input_path": str(extract_dir / "images"),
        "metadata_path": str(extract_dir / "metadata"),
        "metadata_type": dataset_information["metadata_type"],
        "image_type": dataset_information["image_type"],
    }
    if dataset_information.get("append_data_to_metadata"):
        information["append_data_to_metadata"] = str(extract_dir / "metadata" / "appended_metadata.csv"),
    return information
