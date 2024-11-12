"""Helper functions to download and load datasets."""

import hashlib
import json
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
from paidiverpy.utils import initialise_logging

logger = initialise_logging(verbose=2)

# Define a base directory for caching
CACHE_DIR = Path.home() / ".paidiverpy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENCE_FILE = CACHE_DIR / "datasets.json"

DATASET_URLS = {
    "pelagic_csv": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/pelagic_csv.zip",
        "metadata_type": "CSV_FILE",
        "image_type": "BMP",
    },
    "benthic_csv": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_csv.zip",
        "metadata_type": "CSV_FILE",
        "image_type": "PNG",
        "append_data_to_metadata": True,
    },
    "benthic_ifdo": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_ifdo.zip",
        "metadata_type": "IFDO",
        "image_type": "JPG",
    },
}


def load_persistent_paths() -> dict:
    """Load the persistent paths from the cache directory.

    Returns:
        dict: The persistent paths.
    """
    if PERSISTENCE_FILE.exists():
        with PERSISTENCE_FILE.open(encoding="UTF-8") as f:
            return json.load(f)
    return {}


def save_persistent_paths(paths: dict) -> None:
    """Save the persistent paths to the cache directory.

    Args:
        paths (dict): The paths to save.
    """
    with PERSISTENCE_FILE.open("w", encoding="UTF-8") as f:
        json.dump(paths, f)


def download_file(url: str, dataset_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download dataset file from the given URL.

    Download the file from the given URL and cache it locally to avoid redundant downloads.
    A progress bar is displayed for the download process.

    Args:
        url (str): The URL to download the file from.
        cache_dir (Path): The directory to store the downloaded file.
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the downloaded file.
    """
    file_hash = hashlib.md5(url.encode()).hexdigest()
    zip_path = cache_dir / f"{file_hash}.zip"

    if not zip_path.exists():
        logger.info("Downloading %s files...", dataset_name)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Progress bar for downloading
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        with (
            Path.open(zip_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {dataset_name} files",
            ) as bar,
        ):
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))

        logger.info("Downloaded and cached at %s", zip_path)

    return zip_path


def unzip_file(zip_path: Path, dataset_name: str, extract_dir: Path = CACHE_DIR) -> None:
    """Unzip the file to the specified directory.

    Args:
        zip_path (Path): The path to the zip file.
        extract_dir (Path): The directory to extract the contents to.
        dataset_name (str): The name of the dataset.
    """
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            total_files = len(zip_ref.infolist())
            # Progress bar for extraction
            with tqdm(total=total_files, unit="file", desc=f"Extracting {dataset_name} files") as bar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, extract_dir)
                    bar.update(1)
        logger.info("Extracted files to %s", extract_dir)
    else:
        logger.info("Using cached extraction at %s", extract_dir)


def calculate_information(dataset_name: str, extract_dir: Path, dataset_information: dict) -> dict:
    """Calculate the information for the dataset.

    Args:
        dataset_name (str): Dataset name
        extract_dir (Path): Path to the extracted directory
        dataset_information (dict): Information about the dataset

    Returns:
        dict: Information about the dataset
    """
    if dataset_name.split("_")[-1] == "csv":
        metadata_path = f"metadata_{dataset_name}.csv"
    else:
        metadata_path = f"metadata_{dataset_name}.json"
    information = {
        "input_path": str(extract_dir / "images"),
        "metadata_path": str(extract_dir / "metadata" / metadata_path),
        "metadata_type": dataset_information["metadata_type"],
        "image_type": dataset_information["image_type"],
    }
    if dataset_information.get("append_data_to_metadata"):
        information["append_data_to_metadata"] = str(extract_dir / "metadata" / f"appended_metadata_{dataset_name}.csv")

    return information


def load(dataset_name: str) -> dict:
    """Download, unzip, and load the specified dataset.

    Args:
        dataset_name (str): The name of the dataset (for example, 'sample_image').

    Returns:
        dict: A dictionary containing the input path, metadata path, metadata type, and image type.
    """
    dataset_information = DATASET_URLS.get(dataset_name)
    paths = load_persistent_paths()

    if dataset_name in paths and Path(paths[dataset_name]).exists():
        return calculate_information(dataset_name, Path(paths[dataset_name]), dataset_information)
    logger.info("Downloading sample dataset: '%s'", dataset_name)

    extract_dir = CACHE_DIR / dataset_name
    if dataset_information is None:
        msg = f"Dataset '{dataset_name}' not found."
        raise ValueError(msg)
    url = dataset_information["url"]
    zip_path = download_file(url, dataset_name)
    unzip_file(zip_path, dataset_name, extract_dir)

    if zip_path.exists():
        zip_path.unlink()

    paths[dataset_name] = str(extract_dir)
    save_persistent_paths(paths)

    logger.info("Dataset '%s' is available at: %s", dataset_name, extract_dir)

    return calculate_information(dataset_name, extract_dir, dataset_information)
