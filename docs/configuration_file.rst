.. _configuration_file:

Configuration File
==================

The configuration file is a crucial component of the Paidiverpy package. It defines the pipeline you want to run, specifying the input data, processing steps, and output data. Although it is possible to run or create a pipeline without a configuration file, using one is highly recommended to ensure reproducibility and simplify modifications.

Format and Schema
-----------------

The configuration file is written in YAML format and should adhere to the schema detailed in the `configuration file schema <https://github.com/paidiver/paidiverpy/blob/main/src/paidiverpy/configuration-schema.json>`_. Below is an example of a configuration file:

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'
      sampling:
        - mode: 'percent'
          params:
            value: 0.1
      convert:
        - mode: 'bits'
          params:
            output_bits: 8
        - mode: 'to'
          params:
            to: 'gray'

    steps:
      - colour:
          name: 'colour_correction'
          mode: 'grayscale'

      - sampling:
          name: 'datetime'
          mode: 'datetime'
          test: True
          params:
            min: '2016-05-11 04:14:00'
            max: '2016-05-11 09:27:00'

      - colour:
          name: 'colour_correction'
          mode: 'gaussian_blur'
          params:
            sigma: 1.0

      - colour:
          name: 'sharpen'
          mode: 'sharpen'
          params:
            alpha: 1.5
            beta: -0.5

Explanation of the Configuration File
-------------------------------------

In the example above, the configuration file outlines a pipeline with the following steps:


**General Section (Step 0)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains the general information about the pipeline, such as the input and output paths, metadata path, metadata type, image type, and sampling parameters.

- `input_path`: The path to the input images. It can be a local or remote path (e.g., S3 bucket). If it is a private path, you may need to provide object storage credentials. More information on working with remote data, please refer to the :ref:`guide_remote_data`.
- `output_path`: The path to save the output images. Default to **"./output"** It can be a local or remote path (e.g., S3 bucket). You need to have write permissions to this path. More information on working with remote data, please refer to the :ref:`guide_remote_data`.
- `metadata_path`: The path to the metadata file. It needs to be in IFDO standard file (JSON) or CSV file with collumns similar to the IFDO standard. More information on working with metadata, please refer to the :ref:`images_metadata`. It can be a local or remote path (e.g., S3 bucket). More information on working with remote data, please refer to the :ref:`guide_remote_data`.
- `metadata_type`: The type of metadata file. It can be either 'IFDO' or 'CSV'. New types can be added to the package in the future. More information on working with metadata, please refer to the :ref:`images_metadata`.
- `metadata_conventions`: The path to the metadata conventions file. It is used to map the columns of the CSV file to the standard names. If you are using an IFDO file or if you want to use the default conventions, you don't neet to set this parameter. More information on working with metadata, please refer to the :ref:`images_metadata`.
- `image_open_args`: The information about the image type and the parameters to be passed to the image opening function. It can be set to a specific image format (like 'JPG' or 'PNG') or it can have the following keys:
  - `image_type`: The type of images to process. It can be 'JPG', 'PNG', 'TIFF', 'RAW', etc. New types can be added to the package in the future. More information on supporting formats, please refer to the :ref:`guide_image_formats`.
  - `params`: The parameters to be passed to the image opening function. These parameters are specific to each image type and mode. For example, for RAW images, you can specify the width, height, bit depth, etc. More information on opening images, please refer to the :ref:`guide_image_formats`.
- `sampling`: Apply resample to the images in the first step (openning images). In this example, the sampling is set to 10% of the images. More information on sampling images, please refer to the :ref:`step_sampling`.
- `convert`: Apply conversion to the images in the first step (openning images). In this example, the images are converted to 8-bit and grayscale. More information on converting images, please refer to the :ref:`step_convert`.

You can also pass the following parameters to the `general` section:

- `n_jobs`: The number of parallel jobs to run. By default, it is set to 1. If you have a multi-core machine, you can increase this number to speed up the processing. If set to -1, it will use all available cores. More information on parallel processing, please refer to the :ref:`guide_performance`.
- `client`: The Dask client to use for parallel processing. If not provided, it will use the default client. More information on parallel processing, please refer to the :ref:`guide_performance`.
- `track_changes`: If set to `True`, the pipeline will track the changes made to the images at each step. This can be useful for debugging or understanding the processing steps. By default, it is set to `True`.
- `rename`: If set to a value, the output images will be renamed using the specified type. This can be useful for organizing the output images. By default, it is set to `None`. More information on renaming images, please refer to the :ref:`guide_rename_images`.
- `append_data_to_metadata`: It is related to a path of a file with additional metadata to be appended to the metadata file. More information on appending metadata, please refer to the :ref:`images_metadata`.

**Steps Section**
^^^^^^^^^^^^^^^^^

Contains the processing steps to be applied to the images. Each step is defined by a dictionary with the following keys:

- `name`: The name of the processing step. It should correspond to the name of the function in the Paidiverpy package.
- `mode`: The mode of the processing step. It should correspond to the mode of the function in the Paidiverpy package.
- `params`: The parameters to be passed to the processing function. These parameters are specific to each function and mode.
- `test`: A boolean flag to indicate if the step should be run in test mode. If set to `True`, the step will not modify the input images but generate some output graphs or logs. The default value is `False`. For more information, see the :ref:`guide_test_mode` documentation.

In the example above, the pipeline consists of the following steps:

- Step 1. **Colour Conversion**: Convert the images to grayscale.
- Step 2. **Datetime Sampling**: Sample the images based on the datetime metadata.
- Step 3. **Gaussian Blur**: Apply a Gaussian blur with a sigma of 1.0.
- Step 4. **Sharpening**: Sharpen the images using an alpha of 1.5 and a beta of -0.5.

Example Configuration Files
---------------------------

Example configuration files for processing the sample datasets can be found in the `example/config_files <https://github.com/paidiver/paidiverpy/tree/main/examples/config_files>`_ directory of the repository. These files can be used to test the example notebooks described in the :doc:`gallery examples <gallery>`. Running the examples will automatically download the sample data.


.. admonition:: Note

  Some of the examples of configuration file have the flag "sample_data", which is used to indicate that the pipeline will use the sample data. This flag is used in the example notebooks to download the sample data automatically. If you are using your own data, you can remove this flag from the configuration file and update the input path accordingly.


Validation Tools
----------------

To validate your configuration files, you can use the following resources:

- An online validation tool is available: `https://paidiver.github.io/paidiverpy/config_check.html <https://paidiver.github.io/paidiverpy/config_check.html>`_.
- Alternatively, you can validate the configuration file locally using:

1. The command line interface (CLI) of the package:

  .. code-block:: bash

      paidiverpy -c <path_to_config_file> --validate


2. The Python API:

  .. code-block:: python

      from paidiverpy.config.configuration import Configuration

      Configuration.validate_config('<path_to_config_file>', local=False)

If the configuration file is valid, there will be no output. If it is invalid, an error message will be displayed with the details of the validation errors.
