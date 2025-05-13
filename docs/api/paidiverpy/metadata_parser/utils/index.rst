paidiverpy.metadata_parser.utils
================================

.. py:module:: paidiverpy.metadata_parser.utils

.. autoapi-nested-parse::

   Utility functions for metadata parsing.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.metadata_parser.utils.validate_ifdo
   paidiverpy.metadata_parser.utils.format_error


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

.. py:function:: format_error(text: list) -> str

   
   Format error message.

   :param text: List of error messages.
   :type text: list

   :returns: Formatted error message.
   :rtype: str















   ..
       !! processed by numpydoc !!

