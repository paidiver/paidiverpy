paidiverpy.models.step_config
=============================

.. py:module:: paidiverpy.models.step_config

.. autoapi-nested-parse::

   Step configuration module.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.models.step_config.StepConfig
   paidiverpy.models.step_config.PositionConfig
   paidiverpy.models.step_config.ColourConfig
   paidiverpy.models.step_config.ConvertConfig
   paidiverpy.models.step_config.SamplingConfig
   paidiverpy.models.step_config.CustomConfig


Module Contents
---------------

.. py:class:: StepConfig(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Step configuration model.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: model_config
      :type:  ClassVar[dict]

      
      Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].
















      ..
          !! processed by numpydoc !!


   .. py:method:: resolve_params_schema(values: dict) -> dict
      :classmethod:


      
      Resolve the parameters schema based on the step name and mode.

      :param values: The values to validate.
      :type values: dict

      :returns: The validated values.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: update(**updates: dict) -> StepConfig

      
      Update the model in-place with new values.
















      ..
          !! processed by numpydoc !!


.. py:class:: PositionConfig(/, **data: Any)

   Bases: :py:obj:`StepConfig`


   
   Position configuration model.
















   ..
       !! processed by numpydoc !!

.. py:class:: ColourConfig(/, **data: Any)

   Bases: :py:obj:`StepConfig`


   
   Colour configuration model.
















   ..
       !! processed by numpydoc !!

.. py:class:: ConvertConfig(/, **data: Any)

   Bases: :py:obj:`StepConfig`


   
   Convert configuration model.
















   ..
       !! processed by numpydoc !!

.. py:class:: SamplingConfig(/, **data: Any)

   Bases: :py:obj:`StepConfig`


   
   Sampling configuration model.
















   ..
       !! processed by numpydoc !!

.. py:class:: CustomConfig(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Custom configuration model.
















   ..
       !! processed by numpydoc !!

