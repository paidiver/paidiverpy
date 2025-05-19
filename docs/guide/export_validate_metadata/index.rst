.. _guide_export_validate_metadata:

Validate and Export Metadata
============================

In `paidiverpy`, the main standard for metadata is the **iFDO** (image FAIR Digital Object). This standard is designed to provide a consistent and structured way to
store metadata related to images. The iFDO format is a JSON file that contains various attributes describing the image data.

Validating iFDO metadata
------------------------

To ensure compliance with the IFDO standard, it is important to structure the JSON file correctly. You can validate the IFDO metadata by using the `validate_ifdo` function available in the `paidiverpy` package.
This function checks the metadata file against the IFDO standard and ensures that all required fields are present. You can see an example below:

.. code-block:: python

    from paidiverpy.metadata_parser.utils import validate_ifdo

    # Validate the IFDO metadata file
    validate_ifdo("/path/to/your/metadata.json")

This code will output the list of validation errors, if any. This validation also runs when you run a pipeline, because this is a mandatory step before running the pipeline.
Some of the columns in the metadata are not mandatory, but it is recommended to include them to use the full potential of the package.


Exporting metadata
------------------

The package can export the metadata to a CSV file. The CSV file will contain the following columns:
- IFDO format: This is the standard format for metadata in the Paidiverpy package.
- CSV file: You can export the metadata to a CSV file, which can be useful for compatibility with other tools or systems.
- JSON file: You can export the metadata to a JSON file, which is a widely used format for data interchange.
