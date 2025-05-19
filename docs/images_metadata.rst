.. _images_metadata:

Images Metadata
===============

To use this package effectively, you may need a metadata file, which can either be an `IFDO.json` file (adhering to the IFDO standard) or a CSV file. The metadata file plays a crucial role in providing context and additional information about the images you are processing, such as filenames, timestamps, and geospatial coordinates.

IFDO File Format
----------------

The IFDO (Image File Data Object) format is a standardized way of organizing metadata related to images. It typically includes various attributes that describe the image data, including but not limited to:

- **Filename**: The name of the image file.
- **Date and Time**: Timestamps indicating when the image was captured.
- **Geospatial Information**: Latitude and longitude coordinates specifying the location of the image capture.

To ensure compliance with the IFDO standard, it is important to structure the JSON file correctly. You can validate the IFDO metadata by using the `validate_ifdo` function available in the `paidiverpy` package.
This function checks the metadata file against the IFDO standard and ensures that all required fields are present. You can see an example below:

.. code-block:: python

    from paidiverpy.metadata_parser.utils import validate_ifdo

    # Validate the IFDO metadata file
    validate_ifdo("/path/to/your/metadata.json")


Please refer to the :ref:`guide_export_validate_metadata` for more details on how to use this feature.
You can also run an example notebook with this feature by exploring the :ref:`gallery` section.


CSV File Requirements
---------------------

If you opt to use a CSV file, please ensure that the column names adheres to certain standard names. To use the full potential of the package, it is recommended to include the following columns in your CSV file:

- **image-filename**: The name of the image file. **This is the only mandatory column**!
- **ID**: A unique identifier for the image (e.g., index or ID).
- **image-datetime**: The date and time when the image was captured.
- **image-latitude**: The latitude coordinate of the image capture location.
- **image-longitude**: The longitude coordinate of the image capture location.
- **image-depth**: The depth at which the image was captured (if applicable).
- **image-altitude-meters**: The altitude of the camera when the image was captured.
- **image-camera-pitch-degrees**: The pitch angle of the camera when the image was captured.
- **image-camera-roll-degrees**: The roll angle of the camera when the image was captured.

The names of the columns can be different from the ones listed above. The code uses a file to map the names of the columns to the standard names: `metadata_conventions.json <https://github.com/paidiver/paidiverpy/blob/dev/src/paidiverpy/metadata_parser/metadata_conventions.json>`_.
You can use the file provided or you can create your own file to map the columns. If you choose to use your own file, you need to set on the configuration file the path to your file. The path should be set in the `general >> metadata_conventions` part. For example:

.. code-block:: yaml

  general:
    input_path: "/input/data/path/"
    output_path: "/output/data/path/"
    metadata_path: "/metadata/path/metadata.json"
    metadata_type: "IFDO"
    metadata_conventions: "/path/to/your/file.json"

.. admonition:: Note

  You can append additional metadata to the CSV file by providing a path to a separate file containing the extra information. This can be useful for including more detailed attributes or context about the images. Please refer to the `general` section of the :doc:`configuration_file` for more information on appending metadata.


Example Files
-------------

Examples of both CSV and IFDO metadata files are available in the ``example/metadata`` directory. You can refer to these examples to guide the creation of your own metadata files: `Example Metadata Files <https://github.com/paidiver/paidiverpy/tree/dev/examples/metadata>`_

By following these guidelines and utilizing the provided examples, you can ensure that your metadata is well-structured and compatible with the Paidiverpy package, facilitating effective image processing.


Exporting Metadata
------------------

The package provides a function to export the metadata to different formats and standards. The function `export_metadata` allows you to export the metadata to:

- IFDO format: This is the standard format for metadata in the Paidiverpy package.
- CSV file: You can export the metadata to a CSV file, which can be useful for compatibility with other tools or systems.
- JSON file: You can export the metadata to a JSON file, which is a widely used format for data interchange.

Please refer to the :ref:`guide_export_validate_metadata` for more details on how to use this feature and the available options for exporting metadata.
You can also run an example notebook with this feature by exploring the :ref:`gallery` section.
