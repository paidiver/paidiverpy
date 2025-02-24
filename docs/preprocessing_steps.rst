.. _preprocessing_steps:


Preprocessing Steps
===================

The preprocessing steps are the core of the pipeline. They define the operations that will be applied to the input images. The steps are defined in the configuration file under the `steps` key.

This is an example of a configuration file with preprocessing steps:

.. code-block:: yaml

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
      - convert:
          mode: 'bits'
          params:
            output_bits: 8

Each step is a dictionary with the following:

- **step type**: The type of operation to be applied. It can be one of the following:

  - `colour`. Please refer to the :ref:`step_colour` documentation for more details.
  - `sampling`. Please refer to the :ref:`step_sampling` documentation for more details.
  - `convert`. Please refer to the :ref:`step_convert` documentation for more details.
  - `position`. Please refer to the :ref:`step_position` documentation for more details.
  - `custom`. Please refer to the :ref:`step_custom` documentation for more details.


- **name**: A custom name for the step. If not provided, the name will be generated automatically.
- **mode**: The mode of the operation. It will depend on the step type. Please refer to the specific step documentation for more details.
- **params**: Additional parameters specific to the operation. The parameters will depend on the step type and mode. Please refer to the specific step documentation for more details.

The order of the steps in the configuration file determines the order in which they will be applied to the input images. The output of each step will be the input for the next step in the pipeline.

.. admonition:: Note

  The configuration file is written in YAML format and should adhere to the schema detailed in the `configuration file schema <https://github.com/paidiver/paidiverpy/blob/dev/src/paidiverpy/configuration-schema.json>`_. A tool to validate the configuration file against the schema is available `here <https://paidiver.github.io/paidiverpy/config_check.html>`_.


.. toctree::
    :maxdepth: 2

    steps/colour.rst
    steps/sampling.rst
    steps/convert.rst
    steps/position.rst
    steps/custom.rst
