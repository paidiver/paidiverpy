Installation
============

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

3. Install the paidiverpy package:

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

Required and additional dependencies
------------------------------------

Requirement dependencies details can be found `here <https://github.com/paidiver/paidiverpy/blob/dev/pyproject.toml>`_. These dependencies will be installed automatically when you install the package.

You may also need to install the following packages (required by opencv-python):

- libgl
- libegl
- libopengl

On Ubuntu/Debian, you can install these packages using the following command:

.. code-block:: text

  sudo apt install -y libgl1 libegl1 libopengl0
