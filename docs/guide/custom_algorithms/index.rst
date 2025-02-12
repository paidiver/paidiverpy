.. _guide_custom_algorithms:

Custom Algorithm Guide
======================

In `paidiverpy`, you have the flexibility to add your own algorithm to the suite of available algorithms. This guide walks you through the steps to create, implement, and configure a custom algorithm.

Creating a Custom Algorithm
---------------------------

To create a custom algorithm, start by creating a new file that contains a class inheriting from the `BaseCustomAlgorithm` class. This base class is located in the `paidiverpy.custom_layer.base_custom_algorithm` module, shown below:

.. literalinclude:: ../../../src/paidiverpy/custom_layer/base_custom_algorithm.py

Your custom algorithm class should extend `BaseCustomAlgorithm` and implement the `process` method. Here’s a simple example:

.. code-block:: python

  from paidiverpy.custom_layer.base_custom_algorithm import BaseCustomAlgorithm

  class MyMethod(BaseCustomAlgorithm):
      def process(self):
          return self.image_data * self.params.some_param

In the `process` method:

* The input, `image_data`, is either a NumPy or Dask array.
* `params` is an object containing the parameters specific to your algorithm.
* The method should return a processed NumPy or Dask array.

If your algorithm relies on external libraries, import them within this file, ensuring the `process` method follows this signature.

Configuration File
------------------

After creating your custom algorithm, specify it in the configuration file as a pipeline step. Below is an example configuration:

.. code-block:: text

  general:
    # General configurations here

  steps:
    # Steps before the custom algorithm

    - custom:
        name: "my_custom_algorithm"   # Name of the algorithm
        file_path: "/path/to/file.py" # Path to the module implementing the custom algorithm
        class_name: "MyMethod"        # Name of the custom algorithm class
        dependencies:                 # List of dependencies
          - "marimba"
          - "scikit-learn==0.24.2"
        dependencies_path: "/path/to/requirements.txt"  # Optional path to a requirements file
        params:                       # Algorithm parameters
          some_param: 10
          another_param: 0.5

    # Steps following the custom algorithm

In this example:

* The custom algorithm, named `my_custom_algorithm`, is defined in `/path/to/file.py` and implemented in the class `MyMethod`.
* The algorithm parameters include `some_param` (10) and `another_param` (0.5).
* Dependencies are specified both as a list and optionally via a requirements file. Both sets of dependencies are installed before executing the algorithm.

.. admonition:: Important

  You only need to specify external packages as dependencies; packages already available in your environment or included with `paidiverpy` do not need to be listed and will be ignored.

Real Example
------------

For a more concrete example, consider the following code snippet (available in `examples/custom_algorithms files <https://github.com/paidiver/paidiverpy/blob/dev/src/paidiverpy/custom_layer/_custom_algorithm_example.py>`_ of the `paidiverpy` package):

.. literalinclude:: ../../../src/paidiverpy/custom_layer/_custom_algorithm_example.py

In this example, the custom algorithm accepts an image and a `feature_range` parameter. Using `sklearn`'s `MinMaxScaler`, it normalizes the image data within the specified range, then returns the processed data.

The corresponding configuration file might look like this:

.. code-block:: text

  general:
    # General configurations here

  steps:
    # Steps before the custom algorithm

    - custom:
        name: "min_max_data"
        file_path: "/path/to/file.py"
        class_name: "MyMethod"
        dependencies:
          - "scikit-learn"
        params:
          feature_range: (0, 1)

    # Steps following the custom algorithm

In this setup:

* The custom algorithm `min_max_data` resides in `/path/to/file.py`, with the class name `MyMethod`.
* The algorithm has one parameter, `feature_range`, set to `(0, 1)`.
* The dependency `scikit-learn` is installed before the algorithm runs.

To execute, run your application with the configuration file above, and the custom algorithm will be applied accordingly.

Example configuration files for custom algorithms can be found in the `example/config_files <https://github.com/paidiver/paidiverpy/tree/dev/examples/config_files>`_ directory of the repository. You can also run an example notebook with a custom algorithm by exploring the :ref:`gallery` section.

Run in Docker
-------------

To pass the custom algorithm to the Docker container, you need to mount the custom algorithm file to the container. The following steps show how to run the container with a custom algorithm:

.. code-block:: text

  docker run --rm \
  -v <INPUT_PATH>:/app/input/ \
  -v <OUTPUT_PATH>:/app/output/ \
  -v <FULL_PATH_OF_CONFIGURATION_FILE_WITHOUT_FILENAME>:/app/config_files \
  -v <METADATA_PATH_WITHOUT_FILENAME>:/app/metadata/ \
  -v <FULL_PATH_OF_CUSTOM_ALGORITHM_FILE_AND_REQUIREMENTS_FILE>:/app/custom_algorithms \
  paidiverpy \
  paidiverpy -c /app/examples/config_files/<CONFIGURATION_FILE_FILENAME>

In this command:

* `<INPUT_PATH>`: The input path defined in your configuration file, where the input images are located.
* `<OUTPUT_PATH>`: The output path defined in your configuration file.
* `<FULL_PATH_OF_CONFIGURATION_FILE_WITHOUT_FILENAME>`: The local directory of your configuration file.
* `<CONFIGURATION_FILE_FILENAME>`: The name of the configuration file.
* `<FULL_PATH_OF_CUSTOM_ALGORITHM_FILE_AND_REQUIREMENTS_FILE>`: The local directory of your custom algorithm file and requirements file (if any).

The output images will be saved to the specified `output_path`.
