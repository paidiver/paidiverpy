[![DOI][zenodo-badge]][zenodo-link]
[![Documentation][rtd-badge]][rtd-link]
[![Pypi][pip-badge]][pip-link]

[zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.14644007.svg
[zenodo-link]: https://doi.org/10.5281/zenodo.14644007
[rtd-badge]: https://img.shields.io/readthedocs/paidiverpy?logo=readthedocs
[rtd-link]: https://paidiverpy.readthedocs.io/en/latest/?badge=latest
[pip-badge]: https://img.shields.io/pypi/v/paidiverpy
[pip-link]: https://pypi.org/project/paidiverpy/


![Logo](docs/_static/logo_paidiver_docs.png)

**Paidiverpy** is a Python package designed to create pipelines for preprocessing image data for biodiversity analysis.

> **Note:** This package is still in active development, and frequent updates and changes are expected. The API and features may evolve as we continue improving it.


## Documentation

The official documentation is hosted on ReadTheDocs.org: https://paidiverpy.readthedocs.io/

> **Note:** Comprehensive documentation is under construction.

## Installation

To install paidiverpy, run:

 ```bash
pip install paidiverpy
 ```

### Build from Source

You can install `paidiverpy` locally or on a notebook server such as JASMIN or the NOC Data Science Platform (DSP). The following steps are applicable to both environments, but steps 2 and 3 are required if you are using a notebook server.

1. Clone the repository:

   ```bash
   # ssh
   git clone git@github.com:paidiver/paidiverpy.git

   # https
   # git clone https://github.com/paidiver/paidiverpy.git

   cd paidiverpy
   ```

2. (Optional) Create a Python virtual environment to manage dependencies separately from other projects. For example, using `conda`:

   ```bash
   conda init

   # Command to restart the terminal. This command may not be necessary if mamba init has already been successfully run before
   exec bash

   conda env create -f environment.yml
   conda activate Paidiverpy
   ```

3. (Optional) For JASMIN or DSP users, you also need to install the environment in the Jupyter IPython kernel. Execute the following command:

   ```bash
   python -m ipykernel install --user --name Paidiverpy
   ```

4. Install the paidiverpy package:

   Finally, you can install the paidiverpy package:

   ```bash
   pip install -e .
   ```

## Package Organisation

### Configuration File

First, create a configuration file. Example configuration files for processing the sample datasets are available in the `example/config` directory. You can use these files to test the example notebooks described in the [Usage section](#usage). Note that running the examples will automatically download the sample data.

The configuration file should follow the JSON schema described in the [configuration file schema](src/paidiverpy/configuration-schema.json). An online tool to validate configuration files is available [here](https://paidiver.github.io/paidiverpy/config_check.html).

### Metadata

To use this package, you may need a metadata file, which can be an IFDO.json file (following the IFDO standard) or a CSV file. For CSV files, ensure the `filename` column uses one of the following headers: `['image-filename', 'filename', 'file_name', 'FileName', 'File Name']`.

Other columns like datetime, latitude, and longitude should follow these conventions:

- Datetime: `['image-datetime', 'datetime', 'date_time', 'DateTime', 'Datetime']`
- Latitude: `['image-latitude', 'lat', 'latitude_deg', 'latitude', 'Latitude', 'Latitude_deg', 'Lat']`
- Longitude: `['image-longitude', 'lon', 'longitude_deg', 'longitude', 'Longitude', 'Longitude_deg', 'Lon']`

Examples of CSV and IFDO metadata files are in the `example/metadata` directory.

### Layers

The package is organised into multiple layers:

![Package Organisation](docs/_static/paidiver_organisation.jpg)

The `Paidiverpy` class serves as the main container for image processing functions. It manages several subclasses for specific processing tasks: `OpenLayer`, `ConvertLayer`, `PositionLayer`, `ResampleLayer`, and `ColourLayer`.

Supporting classes include:

- `Configuration`: Parses and manages configuration files.
- `Metadata`: Handles metadata.
- `ImagesLayer`: Stores outputs from each image processing step.

The `Pipeline` class integrates all processing steps defined in the configuration file.

## Usage

While comprehensive documentation is forthcoming, you can explore various use cases through sample notebooks in the `examples/example_notebooks` directory:

- [Open and display a configuration file and a metadata file](examples/example_notebooks/config_metadata_example.ipynb)
- [Run processing steps without creating a pipeline](examples/example_notebooks/simple_processing.ipynb)
- [Run a pipeline and interact with outputs](examples/example_notebooks/pipeline.ipynb)
- [Run pipeline steps in test mode](examples/example_notebooks/pipeline_testing_steps.ipynb)
- [Create pipelines programmatically](examples/example_notebooks/pipeline_generation.ipynb)
- [Rerun pipeline steps with modified configurations](examples/example_notebooks/pipeline_interaction.ipynb)
- [Use parallelization with Dask](examples/example_notebooks/pipeline_dask.ipynb)
- [Create a LocalCluster and run a pipeline](examples/example_notebooks/pipeline_cluster.ipynb)
- [Run a pipeline using a public dataset with IFDO metadata](examples/example_notebooks/pipeline_ifdo.ipynb)
- [Run a pipeline using a data on a object store](examples/example_notebooks/pipeline_remote_data.ipynb)
- [Add a custom algorithm to a pipeline](examples/example_notebooks/pipeline_custom_algorithm.ipynb)

### Example Data

If you'd like to manually download example data for testing, you can use the following command:

```python
from paidiverpy import data
data.load(DATASET_NAME)
```

Available datasets:

- pelagic_csv
- benthic_csv
- benthic_ifdo

Example data will be automatically downloaded when running the example notebooks.

### Command-Line Arguments

Pipelines can be executed via command-line arguments. For example:

```bash
paidiverpy -c examples/config_files/config_simple.yaml
```

This runs the pipeline according to the configuration file, saving output images to the directory defined in the `output_path`.

## Docker

You can run **Paidiverpy** using Docker by either building the container locally or pulling a pre-built image from **GitHub Container Registry (GHCR)** or **Docker Hub**.

### Build or Pull the Docker Image

You have three options to obtain the Paidiverpy Docker image:

#### **Option 1: Build the container locally**
Clone the repository and build the image:

```bash
git clone git@github.com:paidiver/paidiverpy.git
cd paidiverpy
docker build -t paidiverpy .
```

#### **Option 2: Pull from Docker Hub**
Fetch the latest image from Docker Hub:

```bash
docker pull soutobias/paidiverpy:latest
docker tag soutobias/paidiverpy:latest paidiverpy:latest
```

#### **Option 3: Pull from GitHub Container Registry (GHCR)**
Fetch the latest image from GitHub:

```bash
docker pull ghcr.io/paidiver/paidiverpy:latest
docker tag ghcr.io/paidiver/paidiverpy:latest paidiverpy:latest
```

### Running the Container

To run the container with local input, output, and metadata directories, use the following command:

```bash
docker run --rm \
  -v <INPUT_PATH>:/app/input/ \
  -v <OUTPUT_PATH>:/app/output/ \
  -v <METADATA_PATH>:/app/metadata/ \
  -v <CONFIG_DIR>:/app/config_files/ \
  paidiverpy -c /app/examples/config_files/<CONFIG_FILE>
```

#### **Arguments Explained**
- `<INPUT_PATH>`: Local directory containing input images (as defined in the configuration file).
- `<OUTPUT_PATH>`: Local directory where processed images will be saved.
- `<METADATA_PATH>`: Local directory containing the metadata file.
- `<CONFIG_DIR>`: Local directory containing the configuration file.
- `<CONFIG_FILE>`: Name of the configuration file.

The processed images will be saved in the `output_path` specified in the configuration file.

### Running with Remote Data (Object Store)

If your input data is stored remotely (e.g., in an object store), you **do not** need to mount local volumes for input data. However, to upload processed images to an object store, you must provide authentication credentials via an environment file.

Use the following command:

```bash
docker run --rm \
  -v <CONFIG_DIR>:/app/config_files/ \
  --env-file .env \
  paidiverpy -c /app/examples/config_files/<CONFIG_FILE>
```

#### **Environment File (`.env`)**
Create a `.env` file with your object store credentials:

```bash
OS_SECRET=your_secret
OS_TOKEN=your_token
OS_ENDPOINT=your_endpoint
```

This will allow Paidiverpy to authenticate and interact with the remote storage system.
