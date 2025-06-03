.. _guide_remote_data:

Remote Data
===========

Overview
--------

**Paidiverpy** supports remote data sources for `input_path`, `output_path`, and `metadata_path` in the configuration file.
This allows you to work with data stored in cloud storage services, such as Amazon S3, without needing to download it locally.
The package can handle both public and private data sources, enabling seamless integration with various data storage solutions.
This section explains how to configure and use remote data sources in your pipeline.

Configuring Remote Data
-----------------------

To work with remote data, specify the data source path in the configuration file. The path can be a public or private URL. If the data source is private, credentials may be required.

### Configuration Parameters

The configuration file supports the following parameters:

- **input_path**: Location of input data.
- **output_path**: Destination for processed data.
- **metadata_path**: Path to the metadata file.

### Public vs. Private Data Sources

#### Public URLs
If the remote location has a public URL (e.g., starting with `https://`), no credentials are needed.

Example:

.. code-block:: yaml

    general:
      input_path: 'https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/lazy_load_benthic/'
      output_path: 's3://bucket-name/output/data/path/'
      metadata_path: "https://paidiver-o.s3-ext.jc.rl.ac.uk/paidiverpy/data/lazy_load_benthic/metadata_ifdo_hf.json"
      metadata_type: 'IFDO'
      image_open_args: 'JPG'

    steps:
      # Define pipeline steps

In this case, you can read data and metadata without authentication. However, credentials are required if saving output to a private S3 bucket (if you set `output_path` to a private S3 location).

#### Private URLs
For private remote locations (e.g., starting with `s3://`), you must provide credentials as environment variables:

- `OS_SECRET`
- `OS_TOKEN`
- `OS_ENDPOINT`

Failure to provide credentials when accessing private data will result in an error.

Example:

.. code-block:: yaml

    general:
      input_path: 's3://bucket-name/input/data/path/'
      output_path: 's3://bucket-name/output/data/path/'
      metadata_path: 's3://bucket-name/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'

    steps:
      # Define pipeline steps

In this case, ensure the necessary environment variables are set before running the pipeline.

### Saving Output Data
At the end of the pipeline, if output saving is enabled, the processed images will be stored in the specified S3 output directory.

.. note::

    For details on the configuration file format, refer to the :ref:`configuration_file` section.
