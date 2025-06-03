paidiverpy.custom_layer.custom_layer
====================================

.. py:module:: paidiverpy.custom_layer.custom_layer

.. autoapi-nested-parse::

   Color layer module.

   This module contains the ColorLayer class for processing the images in the
   color layer.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.custom_layer.custom_layer.CustomLayer


Module Contents
---------------

.. py:class:: CustomLayer(parameters: dict, config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   CustomLayer class.

   Process the images in the custom layer.

   :param parameters: The parameters for the step.
   :type parameters: dict
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
   :param images: The images object.
   :type images: ImagesLayer
   :param paidiverpy: The paidiverpy object.
   :type paidiverpy: Paidiverpy
   :param step_name: The name of the step.
   :type step_name: str
   :param config_index: The index of the configuration.
   :type config_index: int
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int















   ..
       !! processed by numpydoc !!

   .. py:method:: run() -> None

      
      Custom Layer run method.

      Run the custom layer steps on the images based on the configuration
      file or parameters.

      :param add_new_step: Whether to add a new step to the images object.
      :type add_new_step: bool, optional

      Defaults to True.















      ..
          !! processed by numpydoc !!


   .. py:method:: load_custom_algorithm(file_path: str, class_name: str, algorithm_name: str) -> callable

      
      Load a custom algorithm class.

      :param file_path: The file path of the custom algorithm.
      :type file_path: str
      :param class_name: The class name.
      :type class_name: str
      :param algorithm_name: The algorithm name.
      :type algorithm_name: str

      :returns: The custom algorithm class.
      :rtype: class















      ..
          !! processed by numpydoc !!


