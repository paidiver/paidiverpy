paidiverpy.config.config
========================

.. py:module:: paidiverpy.config.config

.. autoapi-nested-parse::

   Configuration module.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.config.config.config_class_mapping


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

   .. py:attribute:: name


   .. py:attribute:: step_name


   .. py:attribute:: sample_data


   .. py:attribute:: output_is_remote


   .. py:attribute:: n_jobs


   .. py:attribute:: client


   .. py:attribute:: track_changes


   .. py:attribute:: rename


   .. py:method:: _define_sample_data(sample_data: str) -> None

      
      Define the sample data.

      :param sample_data: The sample data type
      :type sample_data: str















      ..
          !! processed by numpydoc !!


.. py:class:: PositionConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Position configuration class.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: name


   .. py:attribute:: step_name


   .. py:attribute:: mode


   .. py:attribute:: test


.. py:class:: ConvertConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Convert configuration class.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: name


   .. py:attribute:: step_name


   .. py:attribute:: mode


   .. py:attribute:: test


.. py:class:: ColourConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Colour configuration class.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: name


   .. py:attribute:: step_name


   .. py:attribute:: mode


   .. py:attribute:: test


.. py:class:: SamplingConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Sampling configuration class.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: name


   .. py:attribute:: step_name


   .. py:attribute:: mode


   .. py:attribute:: test


.. py:class:: CustomConfig(**kwargs: dict)

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Sampling configuration class.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: name


   .. py:attribute:: step_name


   .. py:attribute:: file_path


   .. py:attribute:: class_name


   .. py:attribute:: test


.. py:data:: config_class_mapping

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

   .. py:attribute:: general
      :value: None



   .. py:attribute:: steps
      :value: []



   .. py:method:: _load_config_from_file(config_file_path: str) -> None

      
      Load the configuration from a file.

      :param config_file_path: The configuration file path.
      :type config_file_path: str

      :raises FileNotFoundError: file not found.
      :raises yaml.YAMLError: yaml error.















      ..
          !! processed by numpydoc !!


   .. py:method:: _validate_config(config: dict) -> None

      
      Validate the configuration.

      :param config: The configuration.
      :type config: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: _validate_general_config(config_data: dict) -> GeneralConfig

      
      Validate the general configuration.

      :param config_data: The configuration data.
      :type config_data: dict

      :raises ValueError: General configuration is not specified.
      :raises ValueError: General configuration is empty.
      :raises ValueError: Input and output paths are not specified.

      :returns: The general configuration.
      :rtype: GeneralConfig















      ..
          !! processed by numpydoc !!


   .. py:method:: _validate_paths(input_path: str, output_path: str) -> None

      
      Validate the input and output paths.

      :param input_path: Input path.
      :type input_path: str
      :param output_path: Output path.
      :type output_path: str

      :raises ValueError: Input and output paths are not specified.















      ..
          !! processed by numpydoc !!


   .. py:method:: _load_steps(config_data: dict) -> None

      
      Load the steps from the configuration data.

      :param config_data: The configuration data.
      :type config_data: dict

      :raises ValueError: Invalid step name.















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


