paidiverpy-checkpoint
=====================

.. py:module:: paidiverpy-checkpoint

.. autoapi-nested-parse::

   Main class for the paidiverpy package.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy-checkpoint.Paidiverpy


Module Contents
---------------

.. py:class:: Paidiverpy(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, client: dask.distributed.Client | None = None, paidiverpy: Paidiverpy = None, track_changes: bool | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   
   Main class for the paidiverpy package.

   :param config_params: The configuration parameters.
                         It can contain the following keys / attributes:
                         - input_path (str): The path to the input files.
                         - output_path (str): The path to the output files.
                         - image_open_args (str): The type of the images.
                         - metadata_path (str): The path to the metadata file.
                         - metadata_type (str): The type of the metadata file.
                         - track_changes (bool): Whether to track changes.
                         - n_jobs (int): The number of n_jobs.
   :type config_params: dict | ConfigParams, optional
   :param config_file_path: The path to the configuration file.
   :type config_file_path: str, optional
   :param config: The configuration object.
   :type config: Configuration, optional
   :param metadata: The metadata object.
   :type metadata: MetadataParser, optional
   :param images: The images object.
   :type images: ImagesLayer, optional
   :param client: The Dask client object.
   :type client: Client, optional
   :param paidiverpy: The paidiverpy object.
   :type paidiverpy: Paidiverpy, optional
   :param track_changes: Whether to track changes. Defaults to None, which means
                         it will be set to the value of the configuration file.
   :type track_changes: bool
   :param logger: The logger object.
   :type logger: logging.Logger, optional
   :param raise_error: Whether to raise an error.
   :type raise_error: bool, optional
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: run(add_new_step: bool = True) -> paidiverpy.images_layer.ImagesLayer | None

      
      Run the paidiverpy pipeline.

      :param add_new_step: Whether to add a new step. Defaults to True.
      :type add_new_step: bool, optional

      :returns: The images object.
      :rtype: ImagesLayer | None















      ..
          !! processed by numpydoc !!


   .. py:method:: process_images(method: callable, params: dict, custom: bool = False) -> xarray.Dataset

      
      Process the images sequentially.

      Method to process the images sequentially.

      :param method: The method to apply to the images.
      :type method: callable
      :param params: The parameters for the method.
      :type params: dict
      :param custom: Whether the method is a custom method. Defaults to False.
      :type custom: bool, optional

      :returns: A dataset containing the processed images and the metadata.
      :rtype: xr.Dataset















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_output_bands(images: xarray.Dataset, func: callable, custom: bool = False) -> int

      
      Calculate the number of output bands.

      :param images: The input images.
      :type images: xr.Dataset
      :param func: The processing function.
      :type func: callable
      :param custom: Whether the function is a custom function. Defaults to False.
      :type custom: bool, optional

      :returns: The number of output bands.
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: process_dataset(images: list[dask.array.core.Array], method: callable, params: paidiverpy.utils.base_model.BaseModel, custom: bool = False) -> tuple[list[numpy.ndarray], pandas.DataFrame]

      
      Process the images as a dataset.

      :param images: The list of images to process.
      :type images: List[da.core.Array]
      :param method: The method to apply to the images.
      :type method: callable
      :param params: The parameters for the method.
      :type params: BaseModel
      :param custom: Whether the method is a custom method. Defaults to False.
      :type custom: bool, optional

      :returns: A tuple containing the list of processed images and the metadata DataFrame.
      :rtype: tuple[list[np.ndarray], pd.DataFrame]















      ..
          !! processed by numpydoc !!


   .. py:method:: get_metadata(flag: int | None = None, output_format: str | None = None) -> xarray.DataArray

      
      Get the metadata object.

      :param flag: The flag to filter the metadata.
      :type flag: int | None
      :param output_format: The format of the metadata.
      :type output_format: str

      :returns: The metadata object.
      :rtype: xr.DataArray















      ..
          !! processed by numpydoc !!


   .. py:method:: set_metadata(metadata: xarray.DataArray, image_ds: xarray.Dataset | None = None) -> None

      
      Set the metadata.

      :param metadata: The metadata to set.
      :type metadata: xr.DataArray
      :param image_ds: The image dataset.
      :type image_ds: xr.Dataset, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: save_images(step: str | int | None = None, image_format: str = 'png', output_path: str | pathlib.Path | None = None) -> None

      
      Save the images.

      :param step: The step order. Defaults to None.
      :type step: int, optional
      :param image_format: The image format. Defaults to "png".
      :type image_format: str, optional
      :param output_path: The output path. Defaults to None.
      :type output_path: str | Path, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: remove_images() -> None

      
      Remove output images from the output path.
















      ..
          !! processed by numpydoc !!


   .. py:method:: prepare_inputs(image_data: numpy.ndarray, params: paidiverpy.utils.base_model.BaseModel | None, default_params_factory: paidiverpy.utils.base_model.BaseModel, **kwargs: dict) -> tuple[numpy.ndarray, dict, paidiverpy.utils.base_model.BaseModel]
      :staticmethod:


      
      Standard preprocessing for convert layer methods.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters.
      :type params: BaseModel | None
      :param default_params_factory: The default parameters factory.
      :type default_params_factory: BaseModel
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The image data, metadata, and parameters.
      :rtype: tuple[np.ndarray, dict, BaseModel]















      ..
          !! processed by numpydoc !!


   .. py:method:: process_single(img: numpy.ndarray, flag: int, height: int, width: int, metadata: dict, output_bands: int, func: callable, custom: bool) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Wrapper to process a single image with its metadata.

      :param img: The padded image (H, W, bands).
      :type img: np.ndarray
      :param flag: The flag indicating the processing step.
      :type flag: int
      :param height: The height of the valid image area.
      :type height: int
      :param width: The width of the valid image area.
      :type width: int
      :param metadata: The metadata to include.
      :type metadata: dict
      :param output_bands: The number of output bands.
      :type output_bands: int
      :param func: The processing function.
      :type func: callable
      :param custom: Whether to use the custom processing.
      :type custom: bool

      :returns: The processed image (with padding restored) and updated metadata.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


