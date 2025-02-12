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

To ensure compliance with the IFDO standard, it is important to structure the JSON file correctly. You can utilize the `mariqt` package (already integrated to `Paidiverpy`), which provides tools to validate IFDO metadata files against the standard. This ensures that your metadata files are formatted correctly and that all required fields are present.

CSV File Requirements
---------------------

If you opt to use a CSV file, please ensure that the `filename` column adheres to one of the following headers: ['image-filename', 'filename', 'file_name', 'FileName', 'File Name'].

Other columns like datetime, latitude, and longitude should follow these conventions:

- Datetime: ``['image-datetime', 'datetime', 'date_time', 'DateTime', 'Datetime']``
- Latitude: ``['image-latitude', 'lat', 'latitude_deg', 'latitude', 'Latitude', 'Latitude_deg', 'Lat']``
- Longitude: ``['image-longitude', 'lon', 'longitude_deg', 'longitude', 'Longitude', 'Longitude_deg', 'Lon']``

.. admonition:: Note

  You can append additional metadata to the CSV file by providing a path to a separate file containing the extra information. This can be useful for including more detailed attributes or context about the images. Please refer to the `general` section of the :doc:`configuration_file` for more information on appending metadata.


Example Files
-------------

Examples of both CSV and IFDO metadata files are available in the ``example/metadata`` directory. You can refer to these examples to guide the creation of your own metadata files: `Example Metadata Files <https://github.com/paidiver/paidiverpy/tree/dev/examples/metadata>`_

By following these guidelines and utilizing the provided examples, you can ensure that your metadata is well-structured and compatible with the Paidiverpy package, facilitating effective image processing.
