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
   :type config_params: Union[Dict, ConfigParams], optional
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

   .. py:attribute:: step_name


   .. py:attribute:: step_order


   .. py:attribute:: step_metadata


   .. py:method:: run() -> None

      
      Run the resample layer steps on the images based on the configuration.

      Run the resample layer steps on the images based on the configuration.

      :raises ValueError: The mode is not defined in the configuration file.















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_percent(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResamplePercentParams = None) -> pandas.DataFrame

      
      Resample the metadata by a percentage.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResamplePercentParams, optional

      Defaults to ResamplePercentParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_fixed_number(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleFixedParams = None) -> pandas.DataFrame

      
      Resample the metadata by a fixed number of photos.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleFixedParams, optional

      Defaults to ResampleFixedParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_datetime(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleDatetimeParams = None) -> pandas.DataFrame

      
      Resample the metadata by datetime.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleDatetimeParams, optional

      Defaults to ResampleDatetimeParams().

      :raises ValueError: Start date cannot be greater than end date.

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_depth(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleDepthParams = None) -> pandas.DataFrame

      
      Resample the metadata by depth.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleDepthParams, optional

      Defaults to ResampleDepthParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_altitude(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleAltitudeParams = None) -> pandas.DataFrame

      
      Resample the metadata by altitude.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleAltitudeParams, optional

      Defaults to ResampleAltitudeParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_pitch_roll(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResamplePitchRollParams = None) -> pandas.DataFrame

      
      Resample the metadata by pitch and roll.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResamplePitchRollParams, optional

      Defaults to ResamplePitchRollParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_region(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleRegionParams = None) -> pandas.DataFrame

      
      Resample the metadata by region.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleRegionParams, optional

      Defaults to ResampleRegionParams().

      :returns: _description_
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_obscure_photos(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleObscureParams = None) -> pandas.DataFrame

      
      Resample the metadata by obscure photos.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleObscureParams, optional

      Defaults to ResampleObscureParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _by_overlapping(step_order: int | None = None, test: bool = False, params: paidiverpy.config.resample_params.ResampleOverlappingParams = None) -> pandas.DataFrame

      
      Resample the metadata by overlapping photos.

      :param step_order: The order of the step. Defaults to None.
      :type step_order: int, optional
      :param test: Whether to test the step. Defaults to False.
      :type test: bool, optional
      :param params: The parameters for the resample.
      :type params: ResampleOverlappingParams, optional

      Defaults to ResampleOverlappingParams().

      :returns: Metadata with the photos to be removed flagged.
      :rtype: pd.DataFrame















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


