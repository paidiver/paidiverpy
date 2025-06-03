paidiverpy.position_layer.position_layer
========================================

.. py:module:: paidiverpy.position_layer.position_layer

.. autoapi-nested-parse::

   Position layer module.

   Process the images in the position layer.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.position_layer.position_layer.PositionLayer


Module Contents
---------------

.. py:class:: PositionLayer(parameters: dict, config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, add_new_step: bool = True, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Position layer class.

   This class processes the images in the position layer.

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
   :param client: The Dask client.
   :type client: Client
   :param config_index: The index of the configuration.
   :type config_index: int
   :param add_new_step: Whether to add a new step.
   :type add_new_step: bool
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int















   ..
       !! processed by numpydoc !!

   .. py:method:: run() -> None

      
      Run the resample layer steps on the images based on the configuration.

      Run the resample layer steps on the images based on the configuration.

      :raises ValueError: The mode is not defined in the configuration file.















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_corners(step_order: int | None = None, params: paidiverpy.models.position_params.CalculateCornersParams = None, test: bool = False) -> pandas.DataFrame

      
      Calculate the corners of the images.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the position.
      :type params: CalculateCornersParams, optional

      Defaults to CalculateCornersParams().















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_limits(metadata: pandas.DataFrame) -> pandas.DataFrame
      :staticmethod:


      
      Calculate the corners.

      :param metadata: The metadata.
      :type metadata: pd.DataFrame

      :returns: The metadata with the corners.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_corner(lat: float, lon: float, heading_deg: float, headingoffset_rad: float, cornerdist_m: float, angle_offset: float) -> tuple
      :staticmethod:


      
      Calculate the corner coordinates.

      :param lat: The latitude.
      :type lat: float
      :param lon: The longitude.
      :type lon: float
      :param heading_deg: The heading in degrees.
      :type heading_deg: float
      :param headingoffset_rad: The heading offset in radians.
      :type headingoffset_rad: float
      :param cornerdist_m: The corner distance in meters.
      :type cornerdist_m: float
      :param angle_offset: The angle offset.
      :type angle_offset: float

      :returns: The corner coordinates.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


