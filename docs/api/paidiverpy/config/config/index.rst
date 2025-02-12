paidiverpy.config.config
========================

.. py:module:: paidiverpy.config.config

.. autoapi-nested-parse::

   Configuration module.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.config.config.GeneralConfig
   paidiverpy.config.config.PositionConfig
   paidiverpy.config.config.ConvertConfig
   paidiverpy.config.config.ColourConfig
   paidiverpy.config.config.SamplingConfig
   paidiverpy.config.config.CustomConfig
   paidiverpy.config.config.Configuration


Module Contents
---------------

.. py:class:: GeneralConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   General configuration class.

   This class is used to define the general configuration from the configuration file
       or from the input from the user.















   ..
       !! processed by numpydoc !!

.. py:class:: PositionConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Position configuration class.
















   ..
       !! processed by numpydoc !!

.. py:class:: ConvertConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Convert configuration class.
















   ..
       !! processed by numpydoc !!

.. py:class:: ColourConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Colour configuration class.
















   ..
       !! processed by numpydoc !!

.. py:class:: SamplingConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Sampling configuration class.
















   ..
       !! processed by numpydoc !!

.. py:class:: CustomConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Sampling configuration class.
















   ..
       !! processed by numpydoc !!

.. py:class:: Configuration(config_file_path: str | None = None, input_path: str | None = None, output_path: str | None = None)

   
   Configuration class.

   :param config_file_path: The configuration file path. Defaults to None.
   :type config_file_path: str, optional
   :param input_path: The input path. Defaults to None.
   :type input_path: str, optional
   :param output_path: The output path. Defaults to None.
   :type output_path: str, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: add_config(config_name: str, config: dict) -> None

      
      Add a configuration.

      :param config_name: The configuration name.
      :type config_name: str
      :param config: The configuration.
      :type config: dict

      :raises ValueError: Invalid configuration name.















      ..
          !! processed by numpydoc !!


   .. py:method:: add_step(config_index: int | None = None, parameters: dict | None = None) -> int

      
      Add a step to the configuration.

      :param config_index: The configuration index. Defaults to None.
      :type config_index: int, optional
      :param parameters: The parameters for the step. Defaults to None.
      :type parameters: dict, optional

      :raises ValueError: Invalid step index.

      :returns: The step index.
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: export(output_path: str) -> None

      
      Export the configuration to a file.

      :param output_path: The output path.
      :type output_path: str















      ..
          !! processed by numpydoc !!


   .. py:method:: to_dict(yaml_convert: bool = False) -> dict

      
      Convert the configuration to a dictionary.

      :param yaml_convert: Whether to convert the configuration to a yaml format. Defaults to False.
      :type yaml_convert: bool, optional

      :returns: The configuration as a dictionary.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__() -> str

      
      Return the string representation of the configuration.

      :returns: The string representation of the configuration.
      :rtype: str















      ..
          !! processed by numpydoc !!


