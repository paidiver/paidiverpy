paidiverpy.investigation_layer
==============================

.. py:module:: paidiverpy.investigation_layer

.. autoapi-nested-parse::

   
   __init__.py for position_layer module.
















   ..
       !! processed by numpydoc !!


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/paidiverpy/investigation_layer/investigation_layer/index


Classes
-------

.. autoapisummary::

   paidiverpy.investigation_layer.InvestigationLayer


Package Contents
----------------

.. py:class:: InvestigationLayer(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_order: str | None = None, step_name: str | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2, plots: list | str | None = None, plot_metadata: pandas.DataFrame = None)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Investigation layer class.

   This class processes the images in the position layer.

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
   :param parameters: The parameters for the step.
   :type parameters: dict
   :param config_index: The index of the configuration.
   :type config_index: int
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int
   :param plots: The plots to generate.
   :type plots: list | str
   :param plot_metadata: The metadata for the
   :type plot_metadata: pd.DataFrame















   ..
       !! processed by numpydoc !!

   .. py:method:: run() -> None

      
      Run the investigation layer.
















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_trimmed_photos(new_metadata: pandas.DataFrame) -> None

      
      Plot the trimmed photos.

      :param new_metadata: The new metadata.
      :type new_metadata: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_polygons() -> None

      
      Plot the polygons.
















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_brightness_hist(metadata: pandas.DataFrame) -> None

      
      Plot the images.

      :param metadata: The metadata with the images.
      :type metadata: pd.DataFrame















      ..
          !! processed by numpydoc !!


