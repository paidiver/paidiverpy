paidiverpy.pipeline.pipeline
============================

.. py:module:: paidiverpy.pipeline.pipeline

.. autoapi-nested-parse::

   Pipeline builder class for image preprocessing.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.pipeline.pipeline.STEP_WITHOUT_PARAMS
   paidiverpy.pipeline.pipeline.STEP_WITH_PARAMS


Classes
-------

.. autoapisummary::

   paidiverpy.pipeline.pipeline.Pipeline


Module Contents
---------------

.. py:data:: STEP_WITHOUT_PARAMS
   :value: 2


.. py:data:: STEP_WITH_PARAMS
   :value: 3


.. py:class:: Pipeline(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.config.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, steps: list[tuple] | None = None, track_changes: bool | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Pipeline builder class for image preprocessing.

   :param config_params: The configuration parameters.
                         It can contain the following keys / attributes:
                         - input_path (str): The path to the input files.
                         - output_path (str): The path to the output files.
                         - metadata_path (str): The path to the metadata file.
                         - metadata_type (str): The type of the metadata file.
                         - track_changes (bool): Whether to track changes.
                         - n_jobs (int): The number of n_jobs.
   :type config_params: Union[Dict, ConfigParams], optional
   :param config_file_path: The path to the configuration file.
   :type config_file_path: str
   :param config: The configuration object.
   :type config: Configuration
   :param metadata: The metadata object.
   :type metadata: MetadataParser
   :param steps: The steps of the pipeline.
   :type steps: list[tuple], optional
   :param track_changes: Whether to track changes. Defaults to None, which means
                         it will be set to the value of the configuration file.
   :type track_changes: bool
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int















   ..
       !! processed by numpydoc !!

   .. py:attribute:: steps


   .. py:attribute:: runned_steps


   .. py:method:: run(from_step: int | None = None, close_client: bool = True) -> None

      
      Run the pipeline.

      :param from_step: The step to start from. Defaults to None,
                        which means the pipeline will start from the last runned step.
      :type from_step: int, optional
      :param close_client: Whether to close the client. Defaults to True.
      :type close_client: bool, optional

      :raises ValueError: No steps defined for the pipeline
      :raises ValueError: Invalid step format















      ..
          !! processed by numpydoc !!


   .. py:method:: _validate_pipeline() -> None

      
      Validate the pipeline.

      :raises ValueError: No steps defined for the pipeline















      ..
          !! processed by numpydoc !!


   .. py:method:: _validate_from_step(from_step: int | None) -> None

      
      Validate the from_step parameter.
















      ..
          !! processed by numpydoc !!


   .. py:method:: _get_steps_params(step: tuple) -> tuple

      
      Get the parameters of the step.

      :param step: The step.
      :type step: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: export_config(output_path: str) -> None

      
      Export the configuration to a yaml file.

      :param output_path: The path to the output file.
      :type output_path: str















      ..
          !! processed by numpydoc !!


   .. py:method:: add_step(step_name: str, step_class: str | type, parameters: dict, index: int | None = None, substitute: bool = False) -> None

      
      Add a step to the pipeline.

      :param step_name: Name of the step.
      :type step_name: str
      :param step_class: Class of the step.
      :type step_class: Union[str, type]
      :param parameters: Parameters for the step.
      :type parameters: dict
      :param index: Index of the step. It is only used when you
      :type index: int, optional

      want to add a step in a specific position. Defaults to None.
          substitute (bool, optional): Whether to substitute the step in the
      specified index. Defaults to False.















      ..
          !! processed by numpydoc !!


   .. py:method:: _get_step_name(step_class: type) -> str

      
      Get the name of the step class.

      :param step_class: The class of the step.
      :type step_class: type

      :returns: The name of the step class.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: _convert_config_to_steps() -> list[tuple]

      
      Convert the configuration to steps.

      :returns: The steps of the pipeline.
      :rtype: List[tuple]















      ..
          !! processed by numpydoc !!


   .. py:method:: to_html() -> str

      
      Generate HTML representation of the pipeline.

      :returns: The HTML representation of the pipeline.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: _repr_html_() -> str

      
      Generate HTML representation of the pipeline.

      :returns: The HTML representation of the pipeline.
      :rtype: str















      ..
          !! processed by numpydoc !!


