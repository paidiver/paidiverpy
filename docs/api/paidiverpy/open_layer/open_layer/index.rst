paidiverpy.open_layer.open_layer
================================

.. py:module:: paidiverpy.open_layer.open_layer

.. autoapi-nested-parse::

   Open raw image file.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.open_layer.open_layer.OpenLayer


Module Contents
---------------

.. py:class:: OpenLayer(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str = 'raw', parameters: dict | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Open raw image file.

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
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int















   ..
       !! processed by numpydoc !!

   .. py:method:: run() -> None

      
      Run the open layer steps based on the configuration file or parameters.
















      ..
          !! processed by numpydoc !!


   .. py:method:: import_image() -> None

      
      Import images with optional Dask parallelization.
















      ..
          !! processed by numpydoc !!


   .. py:method:: process_image_sequential(img_path: str, remote: bool = False) -> tuple[numpy.ndarray | dask.array.core.Array, dict, str]

      
      Process a single image file.

      :param img_path: The path to the image file
      :type img_path: str
      :param remote: Whether the image is remote. Defaults to False.
      :type remote: bool, optional

      :returns: The processed image, EXIF data, and image path.
      :rtype: np.ndarray | dask.array.core.Array, dict, str















      ..
          !! processed by numpydoc !!


   .. py:method:: rename_images(rename: str, metadata: pandas.DataFrame) -> pandas.DataFrame

      
      Rename images based on the rename mode.

      :param rename: The rename mode
      :type rename: str
      :param metadata: The metadata
      :type metadata: pd.DataFrame

      :raises ValueError: Unknown rename mode

      :returns: The renamed metadata
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


