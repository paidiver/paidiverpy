.. _images_metadata:

Images Metadata
===============


To use this package effectively, you need a metadata file that describes the images being processed.
This can be either an `.json` file (following the iFDO standard) or a CSV file. The metadata provides
essential context such as filenames, timestamps, and geospatial coordinates, which are critical for
accurate image analysis.

Metadata includes both dataset-level information and details for each individual image.

iFDO Standard JSON File
-----------------------

The **iFDO** (image FAIR Digital Object) format is a standardized way to structure metadata for image datasets.
It supports a rich set of attributes that describe both the dataset and individual images, including:

- **Filename**: Name of the image file.
- **Date and Time**: Timestamp of image capture.
- **Geospatial Information**: Latitude and longitude coordinates of the image location.

To ensure iFDO compliance, structure the JSON file according to the standard. You can validate your metadata using the `validate_ifdo` function included in the `paidiverpy` package:

.. code-block:: python

    from paidiverpy.metadata_parser.ifdo_tools import validate_ifdo

    # Validate the iFDO metadata file
    validate_ifdo("/path/to/your/metadata.json")


Please refer to the :ref:`guide_export_validate_metadata` for more details on how to use this feature.
You can also run an example notebook with this feature by exploring the :ref:`gallery` section.


CSV File Requirements
---------------------

If using a CSV file, make sure your column names follow certain conventions. The **`image-filename`** column is mandatory, while the others are optional but recommended for full functionality:

- **image-filename**: The name of the image file. **This is the only mandatory column**!
- **ID**: A unique identifier for the image (e.g., index or ID).
- **image-datetime**: The date and time when the image was captured.
- **image-latitude**: The latitude coordinate of the image capture location.
- **image-longitude**: The longitude coordinate of the image capture location.
- **image-depth**: The depth at which the image was captured (if applicable).
- **image-altitude-meters**: The altitude of the camera when the image was captured.
- **image-camera-pitch-degrees**: The pitch angle of the camera when the image was captured.
- **image-camera-roll-degrees**: The roll angle of the camera when the image was captured.

Column names in your CSV can differ from these standards. The package uses a mapping file, `metadata_conventions.json`, to align your column names with the standard ones:
`metadata_conventions.json <https://github.com/paidiver/paidiverpy/blob/main/src/paidiverpy/metadata_parser/metadata_conventions.json>`_.

You can use the provided mapping file or supply your own by specifying the path in the configuration file:

.. code-block:: yaml

  general:
    input_path: "/input/data/path/"
    output_path: "/output/data/path/"
    metadata_path: "/metadata/path/metadata.json"
    metadata_type: "iFDO"
    metadata_conventions: "/path/to/your/file.json"

.. admonition:: Note

  You can append additional metadata to the CSV file by providing a path to a separate file containing the extra information.
  This can be useful for including more detailed attributes or context about the images. Please refer to the `general` section
  of the :doc:`configuration_file` for more information on appending metadata.

.. admonition:: Note

  The CSV format only supports metadata at the image level. To include dataset-level metadata, use the iFDO format.


Example Files
-------------

Examples of both CSV and iFDO metadata files are available in the on the github repository.
You can refer to these examples to guide the creation of your own metadata files: `Example Metadata Files <https://github.com/paidiver/paidiverpy/tree/main/examples/metadata>`_

EXIF Data
---------

The package can automatically extract **EXIF** (Exchangeable Image File Format) data from image files.
This feature is particularly useful for retrieving metadata such as geospatial coordinates and timestamps
directly embedded within the image.

Once extracted, the EXIF data is seamlessly integrated into the metadata object, enriching it with
additional contextual information without requiring manual input.

Updating Metadata on Pipeline Run
---------------------------------

When you run a pipeline, the package automatically updates the metadata object with new information
generated during the analysis.

This includes adding new attributes or modifying existing ones based on the results of the pipeline steps.

Please refer to the :ref:`preprocessing_steps` section for more details on how the metadata is updated during the pipeline run.

Exporting Metadata
------------------

The package includes an `export_metadata` function for exporting metadata in various formats:

- **iFDO**: The native standard for the Paidiverpy package.
- **CSV**: For use with spreadsheet tools or external systems.
- **JSON**: A flexible, widely supported data format.

Please refer to the :ref:`guide_export_validate_metadata` for more details on how to use this feature and the available options for exporting metadata.
You can also run an example notebook with this feature by exploring the :ref:`gallery` section.
