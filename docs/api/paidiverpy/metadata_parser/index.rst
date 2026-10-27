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

   /api/paidiverpy/metadata_parser/ifdo_tools/index
   /api/paidiverpy/metadata_parser/metadata_parser/index


Classes
-------

.. autoapisummary::

   paidiverpy.metadata_parser.MetadataParser


Package Contents
----------------

.. py:class:: MetadataParser(config: paidiverpy.config.configuration.Configuration | None = None, use_dask: bool = False, metadata_path: str | None = None, metadata_type: str | None = None, metadata_conventions: str | None = None, append_data_to_metadata: str | None = None)

   
   Class for parsing metadata files.

   :param config: Configuration object.
   :type config: Configuration | None
   :param use_dask: Whether to use Dask for parallel processing.
   :type use_dask: bool
   :param metadata_path: Path to the metadata file.
   :type metadata_path: str
   :param metadata_type: Type of the metadata file.
   :type metadata_type: str
   :param append_data_to_metadata: Path to the file with additional data.
   :type append_data_to_metadata: str

   :raises ValueError: Metadata path is not specified.
   :raises ValueError: Metadata type is not specified.















   ..
       !! processed by numpydoc !!

   .. py:method:: open_metadata() -> pandas.DataFrame

      
      Open metadata file.

      :raises ValueError: Metadata type is not supported.

      :returns: Metadata DataFrame.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: set_metadata(metadata: pandas.DataFrame | None = None, dataset_metadata: dict[str, Any] | None = None) -> None

      
      Set the metadata.

      :param metadata: The metadata to set.
      :type metadata: pd.DataFrame | None
      :param dataset_metadata: The dataset metadata to set.
      :type dataset_metadata: dict | None















      ..
          !! processed by numpydoc !!


   .. py:method:: export_metadata(output_format: str = 'csv', output_path: str = 'metadata', metadata: pandas.DataFrame | None = None, dataset_metadata: dict[str, Any] | None = None, from_step: int = -1) -> None

      
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


   .. py:method:: compute() -> None

      
      Compute the metadata if it is a Dask DataFrame.
















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__() -> str

      
      Return the string representation of the metadata.

      :returns: String representation of the metadata.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: convert_metadata_to(dataset_metadata: dict[str, Any], metadata: pandas.DataFrame, output_path: str, output_format: str, from_step: int = -1) -> None
      :staticmethod:


      
      Convert metadata to specified format.

      :param dataset_metadata: Dataset metadata.
      :type dataset_metadata: dict
      :param metadata: Metadata to convert.
      :type metadata: pd.DataFrame
      :param output_path: Path to save the converted metadata.
      :type output_path: str
      :param output_format: Type of metadata to convert to. It can be "csv",
      :type output_format: str

      "json", "IFDO", or "croissant".
          from_step (int): Step to filter metadata. Default is -1, which means the last step.















      ..
          !! processed by numpydoc !!


   .. py:method:: group_metadata_and_dataset_metadata(metadata: pandas.DataFrame, dataset_metadata: dict[str, Any]) -> pandas.DataFrame
      :staticmethod:


      
      Group metadata and dataset metadata.

      :param metadata: Metadata DataFrame.
      :type metadata: pd.DataFrame
      :param dataset_metadata: Dataset metadata.
      :type dataset_metadata: dict
      :param metadata_type: Metadata type. Defaults to "IFDO".
      :type metadata_type: str

      :returns: Combined metadata DataFrame.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


