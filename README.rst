.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14641878.svg
   :target: https://doi.org/10.5281/zenodo.14641878

.. image:: https://img.shields.io/readthedocs/paidiverpy?logo=readthedocs
   :target: https://paidiverpy.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/v/paidiverpy
   :target: https://pypi.org/project/paidiverpy/

.. image:: https://raw.githubusercontent.com/paidiver/paidiverpy/main/docs/_static/logo_paidiver_docs.png

**Paidiverpy** is a Python package designed to create pipelines for preprocessing image data for biodiversity analysis.

.. note::
   This package is still in active development, and frequent updates and changes are expected. The API and features may evolve as we continue improving it.

Documentation
=============

The official documentation is hosted on ReadTheDocs.org: https://paidiverpy.readthedocs.io/

.. note::
   Comprehensive documentation is under construction.

Installation
============

To install paidiverpy, run:

.. code-block:: bash

   pip install paidiverpy

Build from Source
-----------------

You can install `paidiverpy` locally or on a notebook server such as JASMIN or the NOC Data Science Platform (DSP). The following steps are applicable to both environments, but steps 2 and 3 are required if you are using a notebook server.

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

Docker
======

You can run **Paidiverpy** using Docker by either building the container locally or pulling a pre-built image from **GitHub Container Registry (GHCR)** or **Docker Hub**.

Build or Pull the Docker Image
------------------------------

Three options are available:

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

Running the Container
---------------------

.. code-block:: bash

   docker run --rm \
     -v <INPUT_PATH>:/app/input/ \
     -v <OUTPUT_PATH>:/app/output/ \
     -v <METADATA_PATH>:/app/metadata/ \
     -v <CONFIG_DIR>:/app/config_files/ \
     paidiverpy -c /app/examples/config_files/<CONFIG_FILE>
