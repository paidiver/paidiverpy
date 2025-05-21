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

.. py:class:: MetadataParser(config: paidiverpy.config.config.Configuration = None, metadata_path: str | None = None, metadata_type: str | None = None, metadata_conventions: str | None = None, append_data_to_metadata: str | None = None, logger: logging.Logger | None = None)

   
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

   .. py:method:: set_new_attributes(**kwargs: dict) -> None

      
      Set new attributes for the metadata parser.
















      ..
          !! processed by numpydoc !!


   .. py:method:: open_metadata() -> dask.dataframe.DataFrame

      
      Open metadata file.

      :raises ValueError: Metadata type is not supported.

      :returns: Metadata DataFrame.
      :rtype: dd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: export_metadata(output_format: str = 'csv', output_path: str | None = 'metadata', metadata: pandas.DataFrame | None = None, dataset_metadata: dict | None = None, from_step: int = -1) -> None

      
      Export metadata to a file.

      :param output_format: Format of the output file. It can be
      :type output_format: str, optional

      "csv", "json", "IFDO", or "croissant". Defaults to "csv".
          output_path (str, optional): Path to the output file. Defaults to "metadata".
          metadata (pd.DataFrame, optional): Metadata DataFrame. Defaults to None.
          dataset_metadata (dict, optional): Dataset metadata. Defaults to None.
          from_step (int, optional): Step from which to export metadata. Defaults to None, which means last step.















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__() -> str

      
      Return the string representation of the metadata.

      :returns: String representation of the metadata.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: convert_metadata_to(dataset_metadata: dict, metadata: dict, output_path: str, output_format: str, from_step: int = -1) -> None
      :staticmethod:


      
      Convert metadata to specified format.

      :param dataset_metadata: Dataset metadata.
      :type dataset_metadata: dict
      :param metadata: Metadata to convert.
      :type metadata: dict
      :param output_path: Path to save the converted metadata.
      :type output_path: str
      :param output_format: Type of metadata to convert to. It can be "csv",
      :type output_format: str

      "json", "IFDO", or "croissant".
          from_step (int): Step to filter metadata. Default is -1, which means the last step.















      ..
          !! processed by numpydoc !!


