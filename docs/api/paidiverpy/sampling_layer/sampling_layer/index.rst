paidiverpy.sampling_layer.sampling_layer
========================================

.. py:module:: paidiverpy.sampling_layer.sampling_layer

.. autoapi-nested-parse::

   SamplingLayer class.

   Sampling the images based on the configuration file.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.sampling_layer.sampling_layer.SamplingLayer


Module Contents
---------------

.. py:class:: SamplingLayer(parameters: dict[str, Any], config_params: dict[str, Any] | paidiverpy.config.config_params.ConfigParams | None = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration | None = None, metadata: paidiverpy.metadata_parser.MetadataParser | None = None, images: paidiverpy.images_layer.ImagesLayer | None = None, paidiverpy: Optional[paidiverpy.Paidiverpy] = None, step_name: str | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Process the images in the resample layer.

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
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int















   ..
       !! processed by numpydoc !!

   .. py:method:: run(add_new_step: bool = True) -> None | pandas.DataFrame

      
      Run the resample layer steps on the images based on the configuration.

      :param add_new_step: Whether to add a new step. Defaults to True.
      :type add_new_step: bool, optional

      :raises ValueError: The mode is not defined in the configuration file.

      :returns: The result of the resample layer step.
      :rtype: None | pd.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: compute_mean(img: numpy.ndarray[Any, Any], height: int, width: int, bits: int) -> numpy.ndarray[Any, Any]
      :staticmethod:


      
      Compute the mean of the image bands.

      :param img: The input image array.
      :type img: np.ndarray
      :param height: The height of the image.
      :type height: int
      :param width: The width of the image.
      :type width: int
      :param bits: The bit depth of the image.
      :type bits: int

      :returns: The computed mean values for each band.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


