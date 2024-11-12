Installation
============

|License| |Python version| |Anaconda-Server Badge|

|pypi dwn| |conda dwn|

Instructions
------------

You can install the package locally or use our docker image.

- **Install the package locally**:

You can install `paidiverpy` locally or on a notebook server such as JASMIN or the NOC Data Science Platform (DSP). The following steps are applicable to both environments, but steps 2 and 3 are required if you are using a notebook server.

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

3. (Optional) For JASMIN or DSP users, you also need to install the environment in the Jupyter IPython kernel. Execute the following command:

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


.. _Erddapy: https://github.com/ioos/erddapy
.. |Gitter| image:: https://badges.gitter.im/Argo-floats/argopy.svg
   :target: https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |License| image:: https://img.shields.io/badge/License-EUPL%201.2-brightgreen
    :target: https://opensource.org/license/eupl-1-2/
.. |Python version| image:: https://img.shields.io/pypi/pyversions/argopy
   :target: //pypi.org/project/argopy/
.. |Anaconda-Server Badge| image:: https://anaconda.org/conda-forge/argopy/badges/platforms.svg
   :target: https://anaconda.org/conda-forge/argopy
.. |pypi dwn| image:: https://img.shields.io/pypi/dm/argopy?label=Pypi%20downloads
   :target: //pypi.org/project/argopy/
.. |conda dwn| image:: https://img.shields.io/conda/dn/conda-forge/argopy?label=Conda%20downloads
   :target: //anaconda.org/conda-forge/argopy
.. |PyPI| image:: https://img.shields.io/pypi/v/argopy
   :target: //pypi.org/project/argopy/
.. |Conda| image:: https://anaconda.org/conda-forge/argopy/badges/version.svg
   :target: //anaconda.org/conda-forge/argopy
.. |tests in FREE env| image:: https://github.com/euroargodev/argopy/actions/workflows/pytests-free.yml/badge.svg
.. |tests in DEV env| image:: https://github.com/euroargodev/argopy/actions/workflows/pytests-dev.yml/badge.svg
.. |image20| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |image21| image:: https://img.shields.io/github/release-date/euroargodev/argopy
   :target: //github.com/euroargodev/argopy/releases
.. |badge| image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Binder&message=Click+here+to+try+argopy+online+!&color=blue&style=for-the-badge
   :target: https://mybinder.org/v2/gh/euroargodev/binder-sandbox/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Feuroargodev%252Fargopy%26urlpath%3Dlab%252Ftree%252Fargopy%252Fdocs%252Ftryit.ipynb%26branch%3Dmaster
