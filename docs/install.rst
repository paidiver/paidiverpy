Installation
============

|License|

Instructions
------------

You can install the package locally or use our docker image.

- **Install the package locally**:


To install it using pip, you can run the following command:

.. code-block:: text

  pip install paidiverpy

You can also build the package from source. To do so, you need to clone the repository and install the package using the following commands:

1. Clone the repository:

.. code-block:: text

  # ssh
  git clone git@github.com:paidiver/paidiverpy.git

  # you can also clone using https
  # git clone https://github.com/paidiver/paidiverpy.git

  cd paidiverpy


2. (Optional) Create a Python virtual environment to manage dependencies separately from other projects. For example, using `conda`:

.. code-block:: text

  conda init

  # Command to restart the terminal. This command may not be necessary
  # if conda init has already been successfully run before
  exec bash

  conda env create -f environment.yml
  conda activate Paidiverpy

3. (Optional) For notebooks servers, like JASMIN or DSP users, you also need to install the environment in the Jupyter IPython kernel. Execute the following command:

.. code-block:: text

  python -m ipykernel install --user --name Paidiverpy

4. Install the paidiverpy package:

Finally, you can install the paidiverpy package:

.. code-block:: text

  pip install -e .

- **Use the Docker image**:


You can also run Paidiverpy using Docker. You can either build the container locally or pull it from Docker Hub.

1. **Build the container locally**:

.. code-block:: text

  git clone git@github.com:paidiver/paidiverpy.git
  cd paidiverpy
  docker build -t paidiverpy .

2. **Pull the image from Docker Hub**:

.. code-block:: text

  docker pull soutobias/paidiverpy:latest
  docker tag soutobias/paidiverpy:latest paidiverpy:latest

Required dependencies
---------------------

- jsonschema
- mariqt
- opencv
- pillow
- PyYAML
- scikit-image
- scipy
- xarray

Requirement dependencies details can be found `here <https://github.com/paidiver/paidiverpy/blob/develop/pyproject.toml>`_.

These dependencies will be installed automatically when you install the package.

Optional dependencies
---------------------

For a complete **paidiverpy** experience, you may also consider to install the following packages:

**Utilities**

- shapely
- geopy
- geopandas
- tqdm

**Performances**

- dask
- distributed
- dask-image

**Visualisation**

- IPython
- graphviz
- ipykernel
- ipywidgets
- matplotlib


.. |License| image:: https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square
    :target: https://www.apache.org/licenses/
