.. _guide_export_validate_metadata:

Validate and Export Metadata
============================

In `paidiverpy`, metadata is structured using the **iFDO** (image FAIR Digital Object) standard.
iFDO provides a consistent, interoperable format for describing image datasets using JSON.

Validating iFDO metadata
------------------------

To ensure your metadata complies with the iFDO standard, use the `validate_ifdo` function from the `paidiverpy.metadata_parser.ifdo_tools` module.

.. code-block:: python

    from paidiverpy.metadata_parser.ifdo_tools import validate_ifdo

    # Validate the iFDO metadata file
    validate_ifdo("/path/to/your/metadata.json")

This function checks whether the metadata JSON file follows the iFDO schema and includes all required fields and required fields types.
The output of the validation will be a list of errors, if any. If the metadata file is valid, the function will return an empty list.
More details about the iFDO standard can be found in the `iFDO documentation <https://www.marine-imaging.com/fair/ifdos/iFDO-overview/>`_.

.. admonition:: Note

  - Validation is automatically performed when you run a pipeline.
  - Some fields are mandatory per the iFDO specification, while others are required by `paidiverpy` for full pipeline functionality.

Minimum Required Fields (iFDO Standard)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `image-set-header` >> `image-set-ifdo-version`: Version of the iFDO schema used.
- `image-set-items` >> `image-filename`: Name of the image file.


Additional Fields Recommended by `paidiverpy`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the package’s full capabilities, we recommend including the following fields in your metadata file:

- **image-datetime**: The date and time when the image was captured.
- **image-latitude**: The latitude coordinate of the image capture location.
- **image-longitude**: The longitude coordinate of the image capture location.
- **image-depth**: The depth at which the image was captured (if applicable).
- **image-altitude-meters**: The altitude of the camera when the image was captured.
- **image-camera-pitch-degrees**: The pitch angle of the camera when the image was captured.
- **image-camera-roll-degrees**: The roll angle of the camera when the image was captured.


Exporting metadata
------------------


You can export metadata to several formats using the `export_metadata` function inside the `MetadataParser` class.
This is useful for interoperability, archiving, or sharing.

Example Usage
^^^^^^^^^^^^^


.. code-block:: python

    from paidiverpy.pipeline import Pipeline

    pipeline = Pipeline(config_file_path="/path/to/your/config.yaml")
    pipeline.run()

    # Export the metadata to a JSON file
    pipeline.metadata.export_metadata("json", "/path/to/your/output.json")

    # Export the metadata to a CSV file
    pipeline.metadata.export_metadata("csv", "/path/to/your/output.csv")

    # Export the metadata to a iFDO standard file
    pipeline.metadata.export_metadata("ifdo", "/path/to/your/output.json")


from paidiverpy.metadata_parser import MetadataParser

The `export_metadata` function takes the following parameters:

- `output_format` (str): Output format. Options: `"csv"`, `"json"` and `"ifdo"`. Default is `"csv"`.
- `output_path` (str): Path to the output file. Default is `"metadata"`. The file extension will be automatically added based on the `output_format`.
- `metadata` (DataFrame, optional): If not provided, uses pipeline’s internal metadata.
- `dataset_metadata` (dict, optional): Required if original metadata was imported from a CSV. Should match the `image-set-header` structure in the iFDO schema.
- `from_step` (str, optional): Export metadata from a specific pipeline step. If `None`, exports metadata from the final step.

Exported metadata includes:

- Original metadata used when you create the pipeline
- Additional metadata generated during processing
- Extracted EXIF data from images

You can also export metadata using the `MetadataParser` class directly:

.. code-block:: python

    from paidiverpy.metadata.metadata_parser import MetadataParser
    from paidiverpy.config import Config

    config = Config(config_file_path="/path/to/your/config.yaml")
    metadata_parser = MetadataParser(config=config)

    # Export the metadata to a JSON file
    metadata_parser.export_metadata("json", "/path/to/your/output.json")

    # Export the metadata to a CSV file
    metadata_parser.export_metadata("csv", "/path/to/your/output.csv")

    # Export the metadata to a iFDO standard file
    metadata_parser.export_metadata("ifdo", "/path/to/your/output.json")


Supported Output Formats
^^^^^^^^^^^^^^^^^^^^^^^^

**1. CSV**

- Useful for compatibility with spreadsheets or data analysis tools.
- Dataset metadata is added as columns.
- Column names follow iFDO naming conventions where possible.

**2. JSON**

- A flexible, widely supported format.
- Each image’s metadata includes dataset-level metadata as additional keys.

**3. iFDO**

- Exports metadata in the native iFDO JSON format.
- `image-set-header`: Built from `dataset_metadata`
- `image-set-items`: Built from the image-level metadata
- Automatically converts EXIF and pipeline-derived metadata to iFDO fields
- Fills missing required fields with default descriptions from the iFDO schema for easier review and editing

After exporting to iFDO, the file is validated and any schema errors will be printed to the console.

For more information about the iFDO standard, please refer to the `iFDO spec <https://www.marine-imaging.com/fair/ifdos/iFDO-overview/>`_ .
In our :ref:`gallery`, you can find an example of how to use the iFDO standard with the package.
