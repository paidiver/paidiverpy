paidiverpy.metadata_parser
==========================

.. py:module:: paidiverpy.metadata_parser

.. autoapi-nested-parse::

   
   __init__.py for metadata_parser module.
















   ..
       !! processed by numpydoc !!


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/paidiverpy/metadata_parser/metadata_parser/index


Classes
-------

.. autoapisummary::

   paidiverpy.metadata_parser.MetadataParser


Package Contents
----------------

.. py:class:: MetadataParser(config: paidiverpy.config.config.Configuration = None, metadata_path: str | None = None, metadata_type: str | None = None, append_data_to_metadata: str | None = None, logger: logging.Logger | None = None)

   
   Class for parsing metadata files.

   :param config: Configuration object.
   :type config: Configuration
   :param metadata_path: Path to the metadata file.
   :type metadata_path: str
   :param metadata_type: Type of the metadata file.
   :type metadata_type: str
   :param append_data_to_metadata: Path to the file with additional data.
   :type append_data_to_metadata: str
   :param logger: Logger object.
   :type logger: logging.Logger

   :raises ValueError: Metadata path is not specified.
   :raises ValueError: Metadata type is not specified.















   ..
       !! processed by numpydoc !!

   .. py:attribute:: logger


   .. py:attribute:: config


   .. py:attribute:: metadata_type


   .. py:attribute:: append_data_to_metadata


   .. py:attribute:: metadata_path


   .. py:attribute:: storage_options


   .. py:attribute:: metadata


   .. py:attribute:: dataset_metadata
      :value: None



   .. py:method:: _build_config(metadata_path: str, metadata_type: str, append_data_to_metadata: str) -> paidiverpy.config.config.Configuration

      
      Build a configuration object.

      :param metadata_path: Metadata file path.
      :type metadata_path: str
      :param metadata_type: Metadata file type.
      :type metadata_type: str
      :param append_data_to_metadata: Additional data file path.
      :type append_data_to_metadata: str

      :returns: Configuration object.
      :rtype: Configuration















      ..
          !! processed by numpydoc !!


   .. py:method:: open_metadata() -> dask.dataframe.DataFrame

      
      Open metadata file.

      :raises ValueError: Metadata type is not supported.

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _process_coordinates(metadata: dask.dataframe.DataFrame) -> dask.dataframe.DataFrame

      
      Process coordinates in the metadata.

      :param metadata: Metadata DataFrame.
      :type metadata: dd.DataFrame

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _rename_columns(metadata: dask.dataframe.DataFrame, columns: list, raise_error: bool = False) -> dask.dataframe.DataFrame

      
      Rename columns in the metadata.

      :param metadata: Metadata DataFrame.
      :type metadata: dd.DataFrame
      :param columns: List of columns to rename.
      :type columns: list
      :param raise_error: Raise error if column is not found.
      :type raise_error: bool, optional

      Defaults to False.

      :raises ValueError: Metadata does not have a column.

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _add_data_to_metadata(metadata: dask.dataframe.DataFrame) -> dask.dataframe.DataFrame

      
      Add additional data to the metadata.

      :param metadata: Metadata DataFrame.
      :type metadata: dd.DataFrame

      :raises ValueError: Metadata does not have a filename column.

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _open_ifdo_metadata() -> dask.dataframe.DataFrame

      
      Open iFDO metadata file.

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _open_csv_metadata() -> dask.dataframe.DataFrame

      
      Open CSV metadata file.

      :returns: Metadata DataFrame
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _validate_ifdo(ifdo_data: dict) -> None
      :staticmethod:


      
      validate_ifdo method.

      Validates input data against iFDO scheme. Raises an exception if the
      data is invalid.

      :param ifdo_data: parsed iFDO data.
      :type ifdo_data: Dict















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__() -> str

      
      Return the string representation of the metadata.

      :returns: String representation of the metadata.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: _repr_html_() -> str

      
      Return the HTML representation of the metadata.

      :returns: HTML representation of the metadata.
      :rtype: str















      ..
          !! processed by numpydoc !!


