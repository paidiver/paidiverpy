paidiverpy.frontend.parse
=========================

.. py:module:: paidiverpy.frontend.parse

.. autoapi-nested-parse::

   Parse Pydantic models to a dictionary format for frontend use.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.frontend.parse.define_default_value
   paidiverpy.frontend.parse.parse_default_params
   paidiverpy.frontend.parse.parse_get_origin_field
   paidiverpy.frontend.parse.parse_union_field
   paidiverpy.frontend.parse.parse_fields_from_pydantic_model


Module Contents
---------------

.. py:function:: define_default_value(field: pydantic.fields.FieldInfo) -> any

   
   Define the default value for a Pydantic field.

   :param field: The Pydantic field to parse.
   :type field: pydantic.fields.FieldInfo

   :returns: The default value of the field, or None if no default is set.
   :rtype: any















   ..
       !! processed by numpydoc !!

.. py:function:: parse_default_params(model: pydantic.BaseModel, steps: bool = False) -> dict

   
   Parse the default parameters from a Pydantic model.

   :param model: The Pydantic model to parse.
   :type model: pydantic.BaseModel
   :param steps: If True, the parameters will be parsed for steps.
   :type steps: bool

   :returns: A dictionary containing the parsed parameters.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: parse_get_origin_field(field: pydantic.fields.FieldInfo, origin_field: type) -> dict

   
   Parse the origin field of a Pydantic field.

   :param field: The Pydantic field to parse.
   :type field: pydantic.fields.FieldInfo
   :param origin_field: The origin type of the field.
   :type origin_field: type

   :returns: A dictionary containing the parsed field information.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: parse_union_field(field: pydantic.fields.FieldInfo, output: dict) -> dict

   
   Parse a Pydantic field that is a union type.

   :param field: The Pydantic field to parse.
   :type field: pydantic.fields.FieldInfo
   :param output: The output dictionary to populate with parsed information.
   :type output: dict

   :returns: The updated output dictionary with union field information.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: parse_fields_from_pydantic_model(model: pydantic.BaseModel) -> dict

   
   Parse fields from a Pydantic model into a dictionary format.

   :param model: The Pydantic model to parse.
   :type model: pydantic.BaseModel

   :returns: A dictionary containing the parsed fields with their default values and types.
   :rtype: dict















   ..
       !! processed by numpydoc !!

