# Paidiverpy

**Paidiverpy** is a Python package designed to create pipelines for preprocessing image data for biodiversity analysis. 

> **Note:** This package is still in active development, and frequent updates and changes are expected. The API and features may evolve as we continue improving it.

Comprehensive documentation is forthcoming.

## Installation

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

## Package Organization

### Configuration File

First, create a configuration file. Example configuration files for processing the sample datasets are available in the `example/config` directory. You can use these files to test the example notebooks described in the [Usage section](#usage). Note that running the examples will automatically download the sample data.

The configuration file should follow the JSON schema described in the [configuration file schema](configuration-schema.json). An online tool to validate configuration files is available [here](https://paidiver.github.io/paidiverpy/config_check.html).

### Metadata

To use this package, you may need a metadata file, which can be an IFDO.json file (following the IFDO standard) or a CSV file. For CSV files, ensure the `filename` column uses one of the following headers: `['image-filename', 'filename', 'file_name', 'FileName', 'File Name']`.

Other columns like datetime, latitude, and longitude should follow these conventions:
- Datetime: `['image-datetime', 'datetime', 'date_time', 'DateTime', 'Datetime']`
- Latitude: `['image-latitude', 'lat', 'latitude_deg', 'latitude', 'Latitude', 'Latitude_deg', 'Lat']`
- Longitude: `['image-longitude', 'lon', 'longitude_deg', 'longitude', 'Longitude', 'Longitude_deg', 'Lon']`

Examples of CSV and IFDO metadata files are in the `example/metadata` directory.

### Layers

The package is organized into multiple layers:

![Package Organization](docs/images/paidiver_organization.png)

The `Paidiverpy` class serves as the main container for image processing functions. It manages several subclasses for specific processing tasks: `OpenLayer`, `ConvertLayer`, `PositionLayer`, `ResampleLayer`, and `ColorLayer`.

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
- [Run a pipeline using a public dataset with IFDO metadata](examples/example_notebooks/pipeline_ifdo.ipynb)

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

### Docker Command

You can also run Paidiverpy using Docker. You can either build the container locally or pull it from Docker Hub.

1. **Build the container locally**:

    ```bash
    git clone git@github.com:paidiver/paidiverpy.git
    cd paidiverpy
    docker build -t paidiverpy .
    ```

2. **Pull the image from Docker Hub**:

    ```bash
    docker pull soutobias/paidiverpy:latest
    docker tag soutobias/paidiverpy:latest paidiverpy:latest
    ```

Run the container with:

```bash
docker run --rm \
-v <INPUT_PATH>:/app/input/ \
-v <OUTPUT_PATH>:/app/output/ \
-v <FULL_PATH_OF_CONFIGURATION_FILE_WITHOUT_FILENAME>:/app/config_files \
paidiverpy \
paidiverpy -c /app/examples/config_files/<CONFIGURATION_FILE_FILENAME>
```


In this command:
- `<INPUT_PATH>`: The input path defined in your configuration file, where the input images are located.
- `<OUTPUT_PATH>`: The output path defined in your configuration file.
- `<FULL_PATH_OF_CONFIGURATION_FILE_WITHOUT_FILENAME>`: The local directory of your configuration file.
- `<CONFIGURATION_FILE_FILENAME>`: The name of the configuration file.

The output images will be saved to the specified `output_path`.
