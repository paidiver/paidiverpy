.. _package-organisation:

Configuration File
====================

The configuration file is a crucial component of the Paidiverpy package. It defines the pipeline you want to run, specifying the input data, processing steps, and output data. Although it is possible to run or create a pipeline without a configuration file, using one is highly recommended to ensure reproducibility and simplify modifications.

Format and Schema
------------------

The configuration file is written in YAML format and should adhere to the schema detailed in the `configuration file schema <https://github.com/paidiver/paidiverpy/blob/develop/src/paidiverpy/configuration-schema.json>`_. Below is an example of a configuration file:

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_type: 'JPG'
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
------------------

In the example above, the configuration file outlines a pipeline with the following steps:

- Step 0. **Input Processing**: Open the images in the specified input path. The raw images will be converted to 8-bit, and only 10% of the images will be processed.
- Step 1. **Colour Conversion**: Convert the images to grayscale.
- Step 2. **Datetime Sampling**: Sample the images based on the datetime metadata.
- Step 3. **Gaussian Blur**: Apply a Gaussian blur with a sigma of 1.0.
- Step 4. **Sharpening**: Sharpen the images using an alpha of 1.5 and a beta of -0.5.

Example Configuration Files
------------------

Example configuration files for processing the sample datasets can be found in the `example/config_files <https://github.com/paidiver/paidiverpy/tree/develop/examples/config_files>`_ directory of the repository. These files can be used to test the example notebooks described in the :doc:`gallery examples <gallery>`. Running the examples will automatically download the sample data.

Validation Tools
------------------

To validate your configuration files, you can use the following resources:

- An online validation tool is available `here <https://paidiver.github.io/paidiverpy/config_check.html>`_.
- Alternatively, you can validate the configuration file locally using:

.. raw:: html
    :file: config_check.html
