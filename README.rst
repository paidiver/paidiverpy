.. image:: https://raw.githubusercontent.com/paidiver/paidiverpy/main/docs/_static/logo_paidiver_docs.png
    :alt: Paidiverpy

|lifecycle| |License| |Documentation| |DOI| |Pypi|


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

      # ssh
      git clone git@github.com:paidiver/paidiverpy.git

      # https
      # git clone https://github.com/paidiver/paidiverpy.git

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

.. code-block:: text

  # You can export the output images to the specified output directory
  pipeline.save_images(image_format="png")


Command Line Interface (CLI)
----------------------------

Execute the package via the command line.

.. code-block:: bash

    paidiverpy -c "../examples/config_files/config_simple2.yml"

Docker
------

You can run **Paidiverpy** using Docker by either building the container locally or pulling a pre-built image from **GitHub Container Registry (GHCR)** or **Docker Hub**.

1. Build the container locally:

   .. code-block:: bash

      git clone git@github.com:paidiver/paidiverpy.git
      cd paidiverpy
      docker build -t paidiverpy .

2. Pull from GitHub Container Registry (GHCR):

   .. code-block:: bash

      docker pull ghcr.io/paidiver/paidiverpy:latest
      docker tag ghcr.io/paidiver/paidiverpy:latest paidiverpy:latest

3. Pull from Docker Hub:

   .. code-block:: bash

      docker pull soutobias/paidiverpy:latest
      docker tag soutobias/paidiverpy:latest paidiverpy:latest


To run the container, use the following command:

.. code-block:: bash

   docker run --rm \
     -v <INPUT_PATH>:/app/input/ \
     -v <OUTPUT_PATH>:/app/output/ \
     -v <METADATA_PATH>:/app/metadata/ \
     -v <CONFIG_DIR>:/app/config_files/ \
     paidiverpy -c /app/examples/config_files/<CONFIG_FILE>


## Acknowledgements

This project was supported by the UK Natural Environment Research Council (NERC)
through the *Tools for automating image analysis for biodiversity monitoring (AIAB)*
Funding Opportunity, reference code **UKRI052**.


.. |License| image:: https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square
    :target: https://www.apache.org/licenses/
.. .. |Python version| image:: https://img.shields.io/pypi/pyversions/argopy
..    :target: //pypi.org/project/argopy/
.. .. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/argopy/badges/platforms.svg
..    :target: https://anaconda.org/conda-forge/argopy
.. |lifecycle| image:: https://img.shields.io/badge/lifecycle-experimental-green.svg
   :target: https://www.tidyverse.org/lifecycle/#stable
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14641878.svg
   :target: https://doi.org/10.5281/zenodo.14641878
.. |Documentation| image:: https://img.shields.io/readthedocs/paidiverpy?logo=readthedocs
    :target: https://paidiverpy.readthedocs.io/en/latest/?badge=latest
.. |Pypi| image:: https://img.shields.io/pypi/v/paidiverpy
    :target: https://pypi.org/project/paidiverpy/
