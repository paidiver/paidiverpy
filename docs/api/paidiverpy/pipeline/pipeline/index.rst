paidiverpy.pipeline.pipeline
============================

.. py:module:: paidiverpy.pipeline.pipeline

.. autoapi-nested-parse::

   Pipeline builder class for image preprocessing.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.pipeline.pipeline.Pipeline


Module Contents
---------------

.. py:class:: Pipeline(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, steps: list[tuple] | None = None, track_changes: bool | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

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
   :type config_params: dict | ConfigParams, optional
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


   .. py:method:: export_config(output_path: str | None = None) -> None | str

      
      Export the configuration to a yaml file.

      :param output_path: The path to save the configuration file.
      :type output_path: str, optional

      :returns:

                The config file as string if output_path is None,
                    otherwise None.
      :rtype: None | str















      ..
          !! processed by numpydoc !!


   .. py:method:: add_step(step_name: str, step_class: str | type, parameters: dict, index: int | None = None, substitute: bool = False) -> None

      
      Add a step to the pipeline.

      :param step_name: Name of the step.
      :type step_name: str
      :param step_class: Class of the step.
      :type step_class: str | type
      :param parameters: Parameters for the step.
      :type parameters: dict
      :param index: Index of the step. It is only used when you
                    want to add a step in a specific position. Defaults to None.
      :type index: int, optional
      :param substitute: Whether to substitute the step in the
                         specified index. Defaults to False.
      :type substitute: bool, optional















      ..
          !! processed by numpydoc !!


