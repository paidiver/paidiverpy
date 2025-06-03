paidiverpy.config.configuration
===============================

.. py:module:: paidiverpy.config.configuration

.. autoapi-nested-parse::

   Configuration module.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.config.configuration.Configuration


Module Contents
---------------

.. py:class:: Configuration(config_file_path: str | None = None, add_general: dict | None = None, add_steps: list[dict] | None = None)

   
   Configuration class.

   :param config_file_path: The configuration file path. Defaults to None.
   :type config_file_path: str, optional
   :param add_general: The general configuration. Defaults to None.
   :type add_general: dict, optional
   :param add_steps: The steps configuration. Defaults to None.
   :type add_steps: dict, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: validate_config(config: dict | str | pathlib.Path, local: bool = True) -> None
      :staticmethod:


      
      Validate the configuration.

      :param config: The configuration to validate.
      :type config: dict | str | Path
      :param local: Whether the schema is local. Defaults to True.
      :type local: bool, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: add_general(config: dict, validate: bool = False) -> None

      
      Add a configuration.

      :param config: The configuration.
      :type config: dict
      :param validate: Whether to validate the configuration. Defaults to False.
      :type validate: bool, optional

      :raises ValueError: Invalid configuration name.















      ..
          !! processed by numpydoc !!


   .. py:method:: add_step(config_index: int | None = None, parameters: dict | None = None, insert: bool = False, validate: bool = False, step_class: paidiverpy.utils.base_model.BaseModel | None = None) -> int

      
      Add a step to the configuration.

      :param config_index: The configuration index. Defaults to None.
      :type config_index: int, optional
      :param parameters: The parameters for the step. Defaults to None.
      :type parameters: dict, optional
      :param insert: Whether to insert the step at the given index. Defaults to False.
      :type insert: bool, optional
      :param validate: Whether to validate the configuration. Defaults to True.
      :type validate: bool, optional
      :param step_class: The class of the step. Defaults to None.
      :type step_class: BaseModel, optional

      :raises ValueError: Invalid step index.

      :returns: The step index.
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: export(output_path: str | None) -> None | str

      
      Export the configuration to a file.

      :param output_path: The path to save the configuration file. If None, returns the configuration as a YAML string.
      :type output_path: str, optional

      :returns:

                If output_path is None, returns the configuration as a YAML string.
                            Otherwise, writes the configuration to the specified file.
      :rtype: None | str















      ..
          !! processed by numpydoc !!


   .. py:method:: get_output_path(output_path: str | None = None) -> tuple[pathlib.Path | str, bool]

      
      Get the output path.

      :param output_path: The output path. Defaults to None.
      :type output_path: str, optional

      :returns: The output path and whether it is remote.
      :rtype: tuple[Path | str, bool]















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


