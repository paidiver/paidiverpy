paidiverpy.models.general_config
================================

.. py:module:: paidiverpy.models.general_config

.. autoapi-nested-parse::

   Configuration module.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.models.general_config.GeneralConfig


Module Contents
---------------

.. py:class:: GeneralConfig(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   General configuration class.

   This class is used to define the general configuration from the configuration file
       or from the input from the user.















   ..
       !! processed by numpydoc !!

   .. py:attribute:: model_config
      :type:  ClassVar[dict]

      
      Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].
















      ..
          !! processed by numpydoc !!


   .. py:method:: validate_fields(values: dict) -> dict
      :classmethod:


      
      Validate the fields of the configuration.

      :param values: The values to validate.
      :type values: dict

      :returns: The validated values.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: check_required_fields() -> GeneralConfig

      
      Ensure output_path is provided and either sample_data or input_path is set.
















      ..
          !! processed by numpydoc !!


   .. py:method:: update(**updates: dict) -> GeneralConfig

      
      Update the model in-place with new values.
















      ..
          !! processed by numpydoc !!


