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

.. py:function:: validate_ifdo(file_path: str | None = None, ifdo_data: dict | None = None) -> list

   
   validate_ifdo method.

   Validates input data against iFDO scheme. Raises an exception if the
   data is invalid.

   :param file_path: Path to the iFDO file. If not provided, ifdo_data must be.
   :type file_path: str
   :param ifdo_data: parsed iFDO data from the file. If not provided, file_path must be.
   :type ifdo_data: Dict

   :returns: List of validation errors.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: convert_to_ifdo(dataset_metadata: dict, metadata: dict, output_path: str) -> None

   
   Convert metadata to iFDO format.

   :param dataset_metadata: Dataset metadata.
   :type dataset_metadata: dict
   :param metadata: Metadata to convert.
   :type metadata: dict
   :param output_path: Path to save the converted metadata.
   :type output_path: str















   ..
       !! processed by numpydoc !!

.. py:function:: parse_ifdo_items(metadata: dict, ifdo_schema: dict) -> list

   
   Parse iFDO items from metadata.

   :param metadata: Metadata to parse.
   :type metadata: dict
   :param ifdo_schema: iFDO schema.
   :type ifdo_schema: dict

   :returns: Parsed iFDO items.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: parse_ifdo_header(dataset_metadata: dict, ifdo_schema: dict, metadata: pandas.DataFrame) -> tuple

   
   Parse iFDO header from dataset metadata.

   :param dataset_metadata: Dataset metadata.
   :type dataset_metadata: dict
   :param ifdo_schema: iFDO schema.
   :type ifdo_schema: dict
   :param metadata: Metadata to parse.
   :type metadata: pd.DataFrame

   :returns: Parsed iFDO header.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: map_fields_to_ifdo(data: dict, ifdo_data: dict, schema: dict, fields: list, missing_fields: list, missing_fields_suffix: str = '', required: bool = False) -> dict

   
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

.. py:function:: map_exif_to_ifdo(metadata: dict) -> str | None

   
   Map EXIF metadata to iFDO format.

   :param metadata: Metadata to convert.
   :type metadata: dict

   :returns: Converted metadata in iFDO format.
   :rtype: str | None















   ..
       !! processed by numpydoc !!

.. py:function:: format_ifdo_validation_error(text: list) -> str

   
   Format error message.

   :param text: List of error messages.
   :type text: list

   :returns: Formatted error message.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: parse_validation_errors(errors: list, schema: dict) -> list

   
   Parse validation errors.

   :param errors: List of validation errors.
   :type errors: list
   :param schema: JSON schema.
   :type schema: dict

   :returns: Parsed validation errors.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: get_ifdo_fields(schema: dict, section: str) -> tuple

   
   Get required fields from iFDO schema.

   :param schema: JSON schema.
   :type schema: dict
   :param section: Section of the schema to get fields from.
   :type section: str

   :returns: iFDO fields, required fields, non-required fields.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

