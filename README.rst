.. image:: https://raw.githubusercontent.com/paidiver/paidiverpy/main/docs/_static/logo_paidiver_docs.png
    :alt: Paidiverpy

|lifecycle| |License| |Documentation| |DOIlink| |Pypi| |covlink|


**Paidiverpy** is a Python package designed to create pipelines for preprocessing image data for biodiversity analysis.

**IMPORTANT: This package is still in active development, and frequent updates and changes are expected. The API and features may evolve as we continue improving it.**

Documentation
=============

The official documentation is hosted on ReadTheDocs.org: https://paidiverpy.readthedocs.io/

**IMPORTANT: Comprehensive documentation is under construction.**

Installation
============

To install paidiverpy, run:

.. code-block:: bash

   pip install paidiverpy

Build from Source
-----------------

1. Clone the repository:

   .. code-block:: bash

      # https
      git clone https://github.com/paidiver/paidiverpy.git
      cd paidiverpy

2. (Optional) Create a Python virtual environment to manage dependencies separately from other projects. For example, using `conda`:

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate Paidiverpy

3. Install the paidiverpy package:

   .. code-block:: bash

      pip install -e .

Usage
=====

You can run your preprocessing pipeline using **Paidiverpy** in several ways, typically requiring just one to three lines of code:

Python Package
--------------

Install the package and utilize it in your Python scripts.

.. code-block:: text

  # Import the Pipeline class
  from paidiverpy.pipeline import Pipeline

  # Instantiate the Pipeline class with the configuration file path
  # Please refer to the documentation for the configuration file format
  pipeline = Pipeline(config_file_path="../examples/config_files/config_simple2.yml")

  # Run the pipeline
  pipeline.run()

You can export the output images to the specified output directory:

.. code-block:: text

  pipeline.save_images(image_format="png")


Command Line Interface (CLI)
----------------------------

Pipelines can be executed via command-line arguments. For example:

.. code-block:: bash

    paidiverpy -c "examples/config_files/config_simple.yml"

This runs the pipeline according to the configuration file, saving output images to the directory defined in the *output_path*.

Docker
------

You can run **Paidiverpy** using Docker by pulling a pre-built image from **GitHub Container Registry (GHCR)** or **Docker Hub**.


.. code-block:: bash

  docker pull ghcr.io/paidiver/paidiverpy:latest
  docker tag ghcr.io/paidiver/paidiverpy:latest paidiverpy:latest

To run the container, use the following command:

.. code-block:: bash

   docker run --rm \
     -v <INPUT_PATH>:/app/input/ \
     -v <OUTPUT_PATH>:/app/output/ \
     -v <METADATA_PATH>:/app/metadata/ \
     -v <CONFIG_DIR>:/app/config_files/ \
     paidiverpy -c /app/examples/config_files/<CONFIG_FILE>


Example Data
============

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


**IMPORTANT: Please check the documentation for more information about Paidiverpy: https://paidiverpy.readthedocs.io/**

Gallery
=======

Together with the documentation, you can explore various use cases through sample notebooks in the **examples/example_notebooks** directory:


- `Open and display a configuration file and a metadata file <examples/example_notebooks/config_metadata_example.ipynb>`_
- `Run processing steps without creating a pipeline <examples/example_notebooks/simple_processing.ipynb>`_
- `Run a pipeline and interact with outputs <examples/example_notebooks/pipeline.ipynb>`_
- `Run pipeline steps in test mode <examples/example_notebooks/pipeline_testing_steps.ipynb>`_
- `Create pipelines programmatically <examples/example_notebooks/pipeline_generation.ipynb>`_
- `Rerun pipeline steps with modified configurations <examples/example_notebooks/pipeline_interaction.ipynb>`_
- `Use parallelization with Dask <examples/example_notebooks/pipeline_dask.ipynb>`_
- `Create a LocalCluster and run a pipeline <examples/example_notebooks/pipeline_cluster.ipynb>`_
- `Run a pipeline using a public dataset with IFDO metadata <examples/example_notebooks/pipeline_ifdo.ipynb>`_
- `Run a pipeline using a data on a object store <examples/example_notebooks/pipeline_remote_data.ipynb>`_
- `Add a custom algorithm to a pipeline <examples/example_notebooks/pipeline_custom_algorithm.ipynb>`_
- `Open and process raw images <examples/example_notebooks/working_with_raw_images.ipynb>`_
- `Export and validate metadata <examples/example_notebooks/export_validate_metadata.ipynb>`_


Contributing to **paidiverpy**
==============================

Want to support or improve **paidiverpy**? Check out our `contribution guide <https://paidiverpy.readthedocs.io/en/latest/contributing.html>`_ to learn how to get started.


Acknowledgements
================

This project was supported by the UK Natural Environment Research Council (NERC)
through the *Tools for automating image analysis for biodiversity monitoring (AIAB)*
Funding Opportunity, reference code **UKRI052**.

.. |License| image:: https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square
    :target: https://www.apache.org/licenses/
.. .. |Python version| image:: https://img.shields.io/pypi/pyversions/paidiverpy
..    :target: //pypi.org/project/paidiverpy/
.. .. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/paidiverpy/badges/platforms.svg
..    :target: https://anaconda.org/conda-forge/paidiverpy
.. |lifecycle| image:: https://img.shields.io/badge/lifecycle-experimental-green.svg
    :target: https://www.tidyverse.org/lifecycle/#stable
.. |DOIlink| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14641878.svg
    :target: https://doi.org/10.5281/zenodo.14641878
.. |Documentation| image:: https://img.shields.io/readthedocs/paidiverpy?logo=readthedocs
    :target: https://paidiverpy.readthedocs.io/en/latest/?badge=latest
.. |Pypi| image:: https://img.shields.io/pypi/v/paidiverpy
    :target: https://pypi.org/project/paidiverpy/
.. |covlink| image:: https://codecov.io/gh/paidiver/paidiverpy/branch/dev/graph/badge.svg
    :target: https://codecov.io/gh/paidiver/paidiverpy
