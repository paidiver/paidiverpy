paidiverpy
==========

.. py:module:: paidiverpy

.. autoapi-nested-parse::

   
   Paidiverpy base package.
















   ..
       !! processed by numpydoc !!


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/paidiverpy/colour_layer/index
   /api/paidiverpy/config/index
   /api/paidiverpy/convert_layer/index
   /api/paidiverpy/custom_layer/index
   /api/paidiverpy/frontend/index
   /api/paidiverpy/images_layer/index
   /api/paidiverpy/investigation_layer/index
   /api/paidiverpy/metadata_parser/index
   /api/paidiverpy/models/index
   /api/paidiverpy/open_layer/index
   /api/paidiverpy/paidiverpy/index
   /api/paidiverpy/pipeline/index
   /api/paidiverpy/position_layer/index
   /api/paidiverpy/sampling_layer/index
   /api/paidiverpy/utils/index


Classes
-------

.. autoapisummary::

   paidiverpy.Paidiverpy


Functions
---------

.. autoapisummary::

   paidiverpy.show_versions


Package Contents
----------------

.. py:class:: Paidiverpy(config_params: dict[str, Any] | paidiverpy.config.config_params.ConfigParams | None = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration | None = None, metadata: paidiverpy.metadata_parser.MetadataParser | None = None, images: paidiverpy.images_layer.ImagesLayer | None = None, client: dask.distributed.Client | None = None, paidiverpy: Optional[Paidiverpy] = None, track_changes: bool | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   
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


   .. py:method:: process_images(method: collections.abc.Callable, params: dict[str, Any] | paidiverpy.utils.base_model.BaseModel) -> xarray.Dataset

      
      Process the images sequentially.

      Method to process the images sequentially.

      :param method: The method to apply to the images.
      :type method: Callable
      :param params: The parameters for the method.
      :type params: dict | BaseModel

      :returns: A dataset containing the processed images and the metadata.
      :rtype: xr.Dataset















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_output_image(images: xarray.Dataset, func: collections.abc.Callable) -> tuple[dict[str, Any], numpy.dtype[Any]]

      
      Calculate the output image dimensions and data type.

      :param images: The input images.
      :type images: xr.Dataset
      :param func: The processing function.
      :type func: Callable

      :returns: A tuple containing the dask_gufunc_kwargs and the output data type.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: process_dataset(images: xarray.Dataset, method: collections.abc.Callable, params: paidiverpy.utils.base_model.BaseModel) -> xarray.Dataset

      
      Process the images as a dataset.

      :param images: The dataset of images to process.
      :type images: xr.Dataset
      :param method: The method to apply to the images.
      :type method: Callable
      :param params: The parameters for the method.
      :type params: BaseModel

      :returns: A dataset containing the processed images
      :rtype: xr.Dataset















      ..
          !! processed by numpydoc !!


   .. py:method:: get_metadata(flag: int | str | None = None) -> pandas.DataFrame

      
      Get the metadata object.

      :param flag: The flag to filter the metadata.
                   If None, return all metadata. If "all", return all metadata sorted by image-datetime.
                   Defaults to None.
      :type flag: int | str | None, optional

      :returns: The metadata object.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: set_metadata(metadata: pandas.DataFrame | None = None, dataset_metadata: dict[str, Any] | None = None) -> None

      
      Set the metadata.

      :param metadata: The metadata to set.
      :type metadata: pd.DataFrame | None
      :param dataset_metadata: The dataset metadata to set.
      :type dataset_metadata: dict | None















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


   .. py:method:: load_custom_algorithm(file_path: str, class_name: str, algorithm_name: str) -> collections.abc.Callable

      
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


   .. py:method:: process_single(img: numpy.ndarray[Any, Any], flag: int, height: int, width: int, filename: str, output_bands: int | None, func: collections.abc.Callable, metadata: pandas.DataFrame) -> tuple[numpy.ndarray[Any, Any], int, int]
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
      :param filename: The filename of the image.
      :type filename: str
      :param output_bands: The number of output bands.
      :type output_bands: int
      :param func: The processing function.
      :type func: Callable
      :param metadata: The metadata DataFrame.
      :type metadata: pd.DataFrame

      :returns: A tuple containing the processed image, height, and width.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


.. py:function:: show_versions(file: TextIO = sys.stdout, conda: bool = False) -> None

   
   Print the versions of paidiverpy and its dependencies.

   :param file: The file to write the versions to. Defaults to sys.stdout.
   :type file: TextIO, optional
   :param conda: Whether to format the output for conda. Defaults to False.
   :type conda: bool, optional















   ..
       !! processed by numpydoc !!

