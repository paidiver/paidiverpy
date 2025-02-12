paidiverpy.resample_layer.resample_layer
========================================

.. py:module:: paidiverpy.resample_layer.resample_layer

.. autoapi-nested-parse::

   ResampleLayer class.

   Resample the images based on the configuration file.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.resample_layer.resample_layer.ResampleLayer


Module Contents
---------------

.. py:class:: ResampleLayer(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.config.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, parameters: dict | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Process the images in the resample layer.

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
   :param client: The Dask client.
   :type client: Client
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

      
      Run the resample layer steps on the images based on the configuration.

      Run the resample layer steps on the images based on the configuration.

      :raises ValueError: The mode is not defined in the configuration file.















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_polygons(metadata: pandas.DataFrame) -> None
      :staticmethod:


      
      Plot the polygons.

      :param metadata: The metadata with the polygons.
      :type metadata: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_corners(metadata: pandas.DataFrame) -> pandas.DataFrame
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


