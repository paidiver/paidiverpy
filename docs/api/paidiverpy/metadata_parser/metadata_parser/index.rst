paidiverpy.metadata_parser.metadata_parser
==========================================

.. py:module:: paidiverpy.metadata_parser.metadata_parser

.. autoapi-nested-parse::

   Module for parsing metadata files.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.metadata_parser.metadata_parser.MetadataParser


Module Contents
---------------

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

   .. py:method:: open_metadata() -> dask.dataframe.DataFrame

      
      Open metadata file.

      :raises ValueError: Metadata type is not supported.

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__() -> str

      
      Return the string representation of the metadata.

      :returns: String representation of the metadata.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: format_error(text: list) -> str
      :staticmethod:


      
      Format error message.

      :param text: List of error messages.
      :type text: list

      :returns: Formatted error message.
      :rtype: str















      ..
          !! processed by numpydoc !!


