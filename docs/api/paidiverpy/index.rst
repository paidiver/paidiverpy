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
   /api/paidiverpy/static/index
   /api/paidiverpy/utils/index


Classes
-------

.. autoapisummary::

   paidiverpy.Paidiverpy


Package Contents
----------------

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


   .. py:method:: process_sequentially(images: list[numpy.ndarray], method: callable, params: dict, custom: bool = False) -> tuple[list[numpy.ndarray], pandas.DataFrame]

      
      Process the images sequentially.

      Method to process the images sequentially.

      :param images: The list of images to process.
      :type images: List[np.ndarray]
      :param method: The method to apply to the images.
      :type method: callable
      :param params: The parameters for the method.
      :type params: dict
      :param custom: Whether the method is a custom method. Defaults to False.
      :type custom: bool, optional

      :returns: A tuple containing the list of processed images and the metadata DataFrame.
      :rtype: tuple[list[np.ndarray], pd.DataFrame]















      ..
          !! processed by numpydoc !!


   .. py:method:: process_parallel(images: list[dask.array.core.Array], method: callable, params: paidiverpy.utils.base_model.BaseModel, custom: bool = False) -> tuple[list[numpy.ndarray], pandas.DataFrame]

      
      Process the images in parallel.

      Method to process the images in parallel.

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


   .. py:method:: get_metadata(flag: int | None = None) -> pandas.DataFrame

      
      Get the metadata object.

      :param flag: The flag value. Defaults to None.
      :type flag: int, optional

      :returns: The metadata object.
      :rtype: pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: set_metadata(metadata: pandas.DataFrame, flag: bool = False) -> None

      
      Set the metadata.

      :param metadata: The metadata object.
      :type metadata: pd.DataFrame
      :param flag: The flag value. Defaults to False.
      :type flag: bool, optional















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


   .. py:method:: clear_steps(value: int | str) -> None

      
      Clear steps from the images and metadata.

      :param value: Step name or order.
      :type value: int | str















      ..
          !! processed by numpydoc !!


   .. py:method:: prepare_inputs(image_data: numpy.ndarray, metadata: dict | None, params: paidiverpy.utils.base_model.BaseModel | None, default_params_factory: paidiverpy.utils.base_model.BaseModel, **kwargs: dict) -> tuple[numpy.ndarray, dict, paidiverpy.utils.base_model.BaseModel]
      :staticmethod:


      
      Standard preprocessing for convert layer methods.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param metadata: The metadata.
      :type metadata: dict | None
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


