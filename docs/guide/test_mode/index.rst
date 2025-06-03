.. _guide_test_mode:

Test Mode: InvestigationLayer
=============================

Overview
--------

In **paidiverpy**, test mode allows users to execute pipeline steps in a diagnostic mode
using the **InvestigationLayer** class. This is especially useful during debugging, development,
and understanding the behavior of specific processing steps.

When enabled, test mode runs a pipeline step without altering the input images. Instead, it
generates diagnostic outputs such as plots or logs to visualize the operation of the step.

This mode is beneficial for:

- **Understanding** how a specific step operates
- **Non-destructive testing** of pipeline steps
- **Visualisation** of algorithmic effects before application
- **Debugging support** through detailed plots and metadata logging

Enabling Test Mode
------------------

To activate test mode, set the `test` parameter to `True` within the relevant pipeline step in the configuration file.
For example:

.. code-block:: yaml

    general:
      # some general inputs

    steps:
      - colour:
          name: 'colour_correction'
          mode: 'grayscale'
          test: True

In this configuration, the `colour_correction` step will run in test mode, generating logs and
diagnostic plots without modifying the input images.

How Test Mode Works
-------------------

When `test: True` is specified, the step is handled by the **InvestigationLayer** class instead of the standard processing class.
This class:

- Loads metadata and images but skips transformation
- Produces plots such as:

  * Original vs. resampled image locations (`plot_trimmed_photos`)
  * Brightness histograms (`plot_brightness_hist`)
  * Polygon overlays (`plot_polygons`)

- Writes all plots to a dedicated subfolder within the output directory

The type of plot that is generated depends on the step type.

Example Output Structure
^^^^^^^^^^^^^^^^^^^^^^^^

When run with `test: True`, the pipeline creates a directory named after the step, such as:

.. code-block::

    output-path/number_of_step-step_name/
        └── 01_colour_correction/
            ├── graph_trimmed_images.png
            ├── histogram_brightness.png
            └── polygons_overlay.png

Where `output-path` is the specified output_path in the configuration file,
`number_of_step` is the order of the step in the pipeline and `step_name` is the name of the step.


Additional Considerations
-------------------------

- **Output Path**: If the `output_path` is remote, the **InvestigationLayer** will skip plotting and issue a warning.
- **Metadata Requirements**: Some plots require specific metadata fields (e.g., `image-longitude`, `image-latitude`, and others). If these fields are missing, the corresponding plots will not be generated and a warning will be issued.
- **Gallery Examples**: Explore example notebooks with test mode enabled in the :ref:`gallery`.
