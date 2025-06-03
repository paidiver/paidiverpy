[![DOI][zenodo-badge]][zenodo-link]
[![Documentation][rtd-badge]][rtd-link]
[![Pypi][pip-badge]][pip-link]
[![codecov][cov-badge]][cov-link]

[zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.14641878.svg
[zenodo-link]: https://doi.org/10.5281/zenodo.14641878
[rtd-badge]: https://img.shields.io/readthedocs/paidiverpy?logo=readthedocs
[rtd-link]: https://paidiverpy.readthedocs.io/en/latest/?badge=latest
[pip-badge]: https://img.shields.io/pypi/v/paidiverpy
[pip-link]: https://pypi.org/project/paidiverpy/
[cov-badge]: https://codecov.io/gh/paidiver/paidiverpy/branch/dev/graph/badge.svg
[cov-link]: https://codecov.io/gh/paidiver/paidiverpy

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
   conda env create -f environment.yml
   conda activate Paidiverpy
   ```
3. Install the paidiverpy package:

   Finally, you can install the paidiverpy package:

   ```bash
   pip install -e .
   ```


## Usage

You can run your preprocessing pipeline using **Paidiverpy** in several ways, typically requiring just one to three lines of code:


### Python Package

Install the package and utilize it in your Python scripts.

```python
# Import the Pipeline class
from paidiverpy.pipeline import Pipeline

# Instantiate the Pipeline class with the configuration file path
# Please refer to the documentation for the configuration file format
pipeline = Pipeline(config_file_path="../examples/config_files/config_simple2.yml")

# Run the pipeline
pipeline.run()
```

```python
# You can export the output images to the specified output directory
pipeline.save_images(image_format="png")
```


### Command-Line Arguments

Pipelines can be executed via command-line arguments. For example:

```bash
paidiverpy -c examples/config_files/config_simple.yml
```

This runs the pipeline according to the configuration file, saving output images to the directory defined in the `output_path`.


### Gallery

Together with the documentation, you can explore various use cases through sample notebooks in the `examples/example_notebooks` directory:

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
- [Open and process raw images](examples/example_notebooks/working_with_raw_images.ipynb)
- [Export and validate metadata](examples/example_notebooks/export_validate_metadata.ipynb)

### Example Data

If you'd like to manually download example data for testing, you can use the following command:

```python
from paidiverpy.utils.data import PaidiverpyData
PaidiverpyData().load(DATASET_NAME)
```

Available datasets:

- plankton_csv: Plankton dataset with CSV file metadata
- benthic_csv: Benthic dataset with CSV file metadata
- benthic_ifdo: Benthic dataset with IFDO metadata
- nef_raw: Sample images in Nef format (raw images) with CSV file metadata
- benthic_raw_images: Benthic dataset in raw format with CSV file metadata

Example data will be automatically downloaded when running the example notebooks.


> **Note:** Please check the documentation for more information about Paidiverpy: https://paidiverpy.readthedocs.io/

<!--
## 🌍📉 Environmental Impact Report for **paidiverpy**

We care about the environmental footprint of our development process. That’s why we’ve integrated [Green Coding](https://www.green-coding.io/)’s [Green Metrics Tools](https://metrics.green-coding.io) to monitor and estimate the energy consumption and CO₂-equivalent emissions of our GitHub-based CI/CD workflows.

Below are the latest insights from our development activities:

| 🛠️ Activity                      | 🔍 Green Coding Metrics                                                                                                                   |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| CI tests triggered on each commit  | [![CI Energy][ci-energy-badge]][ci-energy-link] [![CI CO₂][ci-energy-badge-co2]][ci-energy-link]                                           |
| Upstream CI tests (daily runs)     | [![Upstream CI Energy][ci-energy-badge-upstream]][ci-energy-link-upstream] [![Upstream CI CO₂][ci-energy-badge-upstream-co2]][ci-energy-link-upstream] |
-->

## Contributing to **paidiverpy**

Want to support or improve **paidiverpy**? Check out our [contribution guide](https://paidiverpy.readthedocs.io/en/latest/contributing.html) to learn how to get started.


## Acknowledgements

This project was supported by the UK Natural Environment Research Council (NERC)
through the *Tools for automating image analysis for biodiversity monitoring (AIAB)*
Funding Opportunity, reference code **UKRI052**.
