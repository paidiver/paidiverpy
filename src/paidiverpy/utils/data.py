"""Helper functions to download and load datasets."""

import hashlib
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any
import requests
from tqdm import tqdm
from paidiverpy.utils.docker import is_running_in_docker

NUM_CHANNELS_GREY = 1
NUM_CHANNELS_RGB = 3
NUM_CHANNELS_RGBA = 4
NUM_DIMENSIONS_GREY = 2
NUM_DIMENSIONS = 3
DEFAULT_BITS = 8
EIGHT_BITS = 8
SIXTEEN_BITS = 16
THIRTY_TWO_BITS = 32
EIGHT_BITS_SIZE = 1
SIXTEEN_BITS_SIZE = 2
THIRTY_TWO_BITS_SIZE = 4
EIGHT_BITS_MAX = 255

# Define a base directory for caching
CACHE_DIR = Path.home() / ".paidiverpy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENCE_FILE = CACHE_DIR / "datasets.json"

DATASET_URLS = {
    "plankton_csv": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/plankton_csv.zip",
        "metadata_type": "CSV_FILE",
        "metadata_path": "metadata_plankton_csv.csv",
        "image_open_args": "BMP",
    },
    "benthic_csv": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_csv.zip",
        "metadata_type": "CSV_FILE",
        "metadata_path": "metadata_benthic_csv.csv",
        "image_open_args": "PNG",
        "append_data_to_metadata": True,
    },
    "benthic_ifdo": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_ifdo.zip",
        "metadata_type": "IFDO",
        "metadata_path": "metadata_benthic_ifdo.json",
        "image_open_args": "JPG",
    },
    "nef_raw": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/nef_raw.zip",
        "metadata_type": "CSV_FILE",
        "metadata_path": "metadata_nef_raw.csv",
        "image_open_args": {
            "image_type": "nef",
            "params": {
                "use_camera_wb": True,
            },
        },
    },
    "benthic_raw_images": {
        "url": "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/benthic_raw_images.zip",
        "metadata_type": "CSV_FILE",
        "metadata_path": "metadata_benthic_raw_images.csv",
        "image_open_args": {
            "image_type": "raw",
            "params": {
                "width": 2448,
                "height": 2048,
                "bit_depth": 8,
                "endianness": None,
                "layout": None,
                "image_misc": "bayer",
                "bayer_pattern": "GB",
                "file_header_size": 0,
            },
        },
    },
}


class PaidiverpyData:
    """A class to download and load datasets."""

    def __init__(self):
        self.logger = logging.getLogger("paidiverpy")

    def load(self, dataset_name: str) -> dict[str, str]:
        """Download, unzip, and load the specified dataset.

        Args:
            dataset_name (str): The name of the dataset (for example, 'sample_image').

        Returns:
            dict: A dictionary containing the input path, metadata path, metadata type, and image type.
        """
        dataset_information = DATASET_URLS.get(dataset_name, {})
        paths = self.load_persistent_paths()

        if dataset_name in paths and Path(paths[dataset_name]).exists():
            return self.calculate_information(dataset_name, Path(paths[dataset_name]), dataset_information)
        self.logger.info("Downloading sample dataset: '%s'", dataset_name)

        extract_dir = CACHE_DIR / dataset_name
        if dataset_information == {}:
            msg = f"Dataset '{dataset_name}' not found."
            raise ValueError(msg)
        url = dataset_information.get("url", "")
        zip_path = self.download_file(str(url), dataset_name)

        self.unzip_file(zip_path, dataset_name, extract_dir)
        if is_running_in_docker():
            self.copy_files_docker(extract_dir, dataset_name)
            extract_dir = Path("/app") / "sample_data" / dataset_name
        paths[dataset_name] = str(extract_dir)

        self.save_persistent_paths(paths)

        self.logger.info("Dataset '%s' is available at: %s", dataset_name, extract_dir)

        return self.calculate_information(dataset_name, extract_dir, dataset_information)

    def copy_files_docker(self, extract_dir: Path, dataset_name: str) -> None:
        """Copy files from the extract directory to the appropriate location in the Docker container.

        Args:
            extract_dir (Path): The directory where the dataset has been extracted.
            dataset_name (str): The name of the dataset.
        """
        metadata_path = Path("/app") / "sample_data" / dataset_name / "metadata"
        images_path = Path("/app") / "sample_data" / dataset_name / "input"
        if metadata_path.exists():
            shutil.rmtree(metadata_path)
        if images_path.exists():
            shutil.rmtree(images_path)
        shutil.copytree(extract_dir / "metadata", metadata_path)
        shutil.copytree(extract_dir / "images", images_path)

    def load_persistent_paths(self) -> dict[str, str]:
        """Load the persistent paths from the cache directory.

        Returns:
            dict[str, str]: The persistent paths.
        """
        if PERSISTENCE_FILE.exists():
            with PERSISTENCE_FILE.open(encoding="UTF-8") as f:
                return json.load(f)
        return {}

    def save_persistent_paths(self, paths: dict[str, str]) -> None:
        """Save the persistent paths to the cache directory.

        Args:
            paths (dict): The paths to save.
        """
        with PERSISTENCE_FILE.open("w", encoding="UTF-8") as f:
            json.dump(paths, f)

    def download_file(self, url: str, dataset_name: str, cache_dir: Path = CACHE_DIR) -> Path:
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
        file_hash = hashlib.sha256(url.encode()).hexdigest()
        zip_path = cache_dir / f"{file_hash}.zip"

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading %s files...", dataset_name)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

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

        self.logger.info("Downloaded and cached at %s", zip_path)

        return zip_path

    def unzip_file(self, zip_path: Path, dataset_name: str, extract_dir: Path = CACHE_DIR) -> None:
        """Unzip the file to the specified directory.

        Args:
            zip_path (Path): The path to the zip file.
            extract_dir (Path): The directory to extract the contents to.
            dataset_name (str): The name of the dataset.
        """
        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    total_files = len(zip_ref.infolist())
                    with tqdm(
                        total=total_files,
                        unit="file",
                        desc=f"Extracting {dataset_name} files",
                    ) as bar:
                        for file_info in zip_ref.infolist():
                            zip_ref.extract(file_info, extract_dir)
                            bar.update(1)
                self.logger.info("Extracted files to %s", extract_dir)
                zip_path.unlink()
            except Exception as e:  # noqa: BLE001
                self.logger.error("Failed to extract files to %s: %s", extract_dir, e)
                self.logger.error("Please try again.")
                zip_path.unlink()
        else:
            self.logger.info("Using cached extraction at %s", extract_dir)

    def calculate_information(self, dataset_name: str, extract_dir: Path, dataset_information: dict[str, Any]) -> dict[str, str]:
        """Calculate the information for the dataset.

        Args:
            dataset_name (str): Dataset name
            extract_dir (Path): Path to the extracted directory
            dataset_information (dict): Information about the dataset

        Returns:
            dict: Information about the dataset
        """
        metadata_path = dataset_information["metadata_path"]
        information = {
            "input_path": str(extract_dir / "input") if is_running_in_docker() else str(extract_dir / "images"),
            "metadata_path": str(extract_dir / "metadata" / metadata_path),
            "metadata_type": dataset_information["metadata_type"],
            "image_open_args": dataset_information["image_open_args"],
        }
        if dataset_information.get("append_data_to_metadata"):
            information["append_data_to_metadata"] = str(extract_dir / "metadata" / f"appended_metadata_{dataset_name}.csv")

        return information
