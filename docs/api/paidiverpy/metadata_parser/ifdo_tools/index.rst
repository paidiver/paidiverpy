paidiverpy.metadata_parser.ifdo_tools
=====================================

.. py:module:: paidiverpy.metadata_parser.ifdo_tools

.. autoapi-nested-parse::

   Utility functions for metadata parsing.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.metadata_parser.ifdo_tools.validate_ifdo
   paidiverpy.metadata_parser.ifdo_tools.convert_to_ifdo
   paidiverpy.metadata_parser.ifdo_tools.parse_ifdo_items
   paidiverpy.metadata_parser.ifdo_tools.parse_ifdo_header
   paidiverpy.metadata_parser.ifdo_tools.map_fields_to_ifdo
   paidiverpy.metadata_parser.ifdo_tools.map_exif_to_ifdo
   paidiverpy.metadata_parser.ifdo_tools.format_ifdo_validation_error
   paidiverpy.metadata_parser.ifdo_tools.parse_validation_errors
   paidiverpy.metadata_parser.ifdo_tools.get_ifdo_fields


Module Contents
---------------

.. py:function:: validate_ifdo(file_path: str | None = None, ifdo_data: dict[str, Any] | None = None) -> list[dict[str, Any]]

   
   validate_ifdo method.

   Validates input data against iFDO scheme. Raises an exception if the
   data is invalid.

   :param file_path: Path to the iFDO file. If not provided, ifdo_data must be.
   :type file_path: str
   :param ifdo_data: parsed iFDO data from the file. If not provided, file_path must be.
   :type ifdo_data: dict

   :returns: List of validation errors.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: convert_to_ifdo(dataset_metadata: dict[str, Any], metadata: pandas.DataFrame, output_path: str) -> None

   
   Convert metadata to iFDO format.

   :param dataset_metadata: Dataset metadata.
   :type dataset_metadata: dict
   :param metadata: Metadata to convert.
   :type metadata: pd.DataFrame
   :param output_path: Path to save the converted metadata.
   :type output_path: str















   ..
       !! processed by numpydoc !!

.. py:function:: parse_ifdo_items(metadata: pandas.DataFrame, ifdo_schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]

   
   Parse iFDO items from metadata.

   :param metadata: Metadata to parse.
   :type metadata: pd.DataFrame
   :param ifdo_schema: iFDO schema.
   :type ifdo_schema: dict

   :returns: Parsed iFDO items and list of missing fields.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: parse_ifdo_header(dataset_metadata: dict[str, Any], ifdo_schema: dict[str, Any], metadata: pandas.DataFrame) -> tuple[dict[str, Any], list[str]]

   
   Parse iFDO header from dataset metadata.

   :param dataset_metadata: Dataset metadata.
   :type dataset_metadata: dict
   :param ifdo_schema: iFDO schema.
   :type ifdo_schema: dict
   :param metadata: Metadata to parse.
   :type metadata: pd.DataFrame

   :returns: Parsed iFDO header and list of missing fields.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: map_fields_to_ifdo(data: dict[str, Any], ifdo_data: dict[str, Any], schema: dict[str, Any], fields: list[str] | set[str], missing_fields: list[str], missing_fields_suffix: str = '', required: bool = False) -> dict[str, Any]

   
   Map fields from dataset metadata to iFDO header.

   :param data: Dataset metadata.
   :type data: dict
   :param ifdo_data: iFDO data to populate.
   :type ifdo_data: dict
   :param schema: iFDO schema.
   :type schema: dict
   :param fields: List of fields to map.
   :type fields: list
   :param missing_fields: List of missing fields.
   :type missing_fields: list
   :param missing_fields_suffix: Suffix to append to missing fields.
   :type missing_fields_suffix: str
   :param required: Whether the fields are required.
   :type required: bool

   :returns: Mapped iFDO header.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: map_exif_to_ifdo(metadata: dict[str, Any]) -> str | None | dict[str, Any]

   
   Map EXIF metadata to iFDO format.

   :param metadata: Metadata to convert.
   :type metadata: dict

   :returns: Converted metadata in iFDO format.
   :rtype: str | None















   ..
       !! processed by numpydoc !!

.. py:function:: format_ifdo_validation_error(text: list[str]) -> str

   
   Format error message.

   :param text: List of error messages.
   :type text: list

   :returns: Formatted error message.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: parse_validation_errors(errors: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]

   
   Parse validation errors.

   :param errors: List of validation errors.
   :type errors: list
   :param schema: JSON schema.
   :type schema: dict

   :returns: Parsed validation errors.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: get_ifdo_fields(schema: dict[str, Any], section: str) -> tuple[dict[str, Any], list[str], set[str]]

   
   Get required fields from iFDO schema.

   :param schema: JSON schema.
   :type schema: dict
   :param section: Section of the schema to get fields from.
   :type section: str

   :returns: iFDO fields, required fields, non-required fields.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

