paidiverpy.utils.schema_json_handler
====================================

.. py:module:: paidiverpy.utils.schema_json_handler

.. autoapi-nested-parse::

   Schema JSON handler module.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.schema_json_handler.generate_schema
   paidiverpy.utils.schema_json_handler.wrap_ref


Module Contents
---------------

.. py:function:: generate_schema(output_path: str) -> None

   
   Generate the schema for the configuration model.

   :param output_path: The path to save the schema.
   :type output_path: str















   ..
       !! processed by numpydoc !!

.. py:function:: wrap_ref(schema: dict, target_ref: str, wrapper_key: str) -> dict

   
   Wrap a reference in the schema with a key.

   :param schema: The original schema.
   :type schema: dict
   :param target_ref: The reference to wrap.
   :type target_ref: str
   :param wrapper_key: The key to wrap the reference with.
   :type wrapper_key: str

   :returns: The modified schema with the reference wrapped.
   :rtype: dict















   ..
       !! processed by numpydoc !!

