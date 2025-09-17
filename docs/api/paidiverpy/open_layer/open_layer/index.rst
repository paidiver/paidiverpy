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

.. py:class:: OpenLayer(config_params: dict[str, Any] | paidiverpy.config.config_params.ConfigParams | None = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration | None = None, metadata: paidiverpy.metadata_parser.MetadataParser | None = None, images: paidiverpy.images_layer.ImagesLayer | None = None, paidiverpy: paidiverpy.Paidiverpy | None = None, step_name: str = 'raw', client: dask.distributed.Client | None = None, parameters: dict[str, Any] | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

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
   :param client: The Dask client.
   :type client: Client
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


   .. py:method:: process_single_image(img_path: str | pathlib.Path, func: collections.abc.Callable, metadata: xarray.DataArray, rename: str, image_type: str, image_open_args: dict[str, Any], storage_options: dict[str, Any]) -> tuple[numpy.ndarray[Any, Any] | dask.array.core.Array, dict[str, Any], str] | None
      :staticmethod:


      
      Process a single image.

      :param img_path: The path to the image.
      :type img_path: str | Path
      :param func: The function to process the image.
      :type func: Callable
      :param metadata: The metadata DataArray.
      :type metadata: xr.DataArray
      :param rename: The rename strategy.
      :type rename: str
      :param image_type: The image type.
      :type image_type: str
      :param image_open_args: The image open arguments.
      :type image_open_args: dict
      :param storage_options: The storage options.
      :type storage_options: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: create_dataset(images_info: dict[str, Any]) -> xarray.Dataset

      
      Create a Dask array from the processed images and EXIF data.

      :param images_info: A dictionary containing processed image information.
      :type images_info: dict

      :returns: The image dataset.
      :rtype: xr.Dataset















      ..
          !! processed by numpydoc !!


