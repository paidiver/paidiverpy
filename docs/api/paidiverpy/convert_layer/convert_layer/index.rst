paidiverpy.convert_layer.convert_layer
======================================

.. py:module:: paidiverpy.convert_layer.convert_layer

.. autoapi-nested-parse::

   Convert Layer.

   Convert the images in the convert layer based on the configuration file or
   parameters.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.convert_layer.convert_layer.ConvertLayer


Module Contents
---------------

.. py:class:: ConvertLayer(parameters: dict, config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Process the images in the convert layer.

   This class provides various methods to convert images according to specified
   configurations, such as resizing, normalizing, bit depth conversion, and channel conversion.

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

   .. py:method:: convert_bits(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.convert_params.BitParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Convert the image to the specified number of bits.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param metadata: The metadata for the image.
      :type metadata: dict, optional
      :param params: The parameters for the bit conversion.
      :type params: BitParams, optional
      :param \*\*kwargs: Additional keyword arguments.

      Defaults to BitParams().

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: channel_convert(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.convert_params.ToParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Convert the image to the specified channel.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param metadata: The metadata for the image.
      :type metadata: dict, optional
      :param params: The parameters for the channel conversion.
                     Defaults to ToParams().
      :type params: ToParams, optional
      :param \*\*kwargs: Additional keyword arguments.

      :raises ValueError: The image is already in RGB format.
      :raises ValueError: The image is already in grayscale.
      :raises ValueError: Failed to convert the image to {params.to}: {str(e)}

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: normalize_image(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.convert_params.NormalizeParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Normalize the image data.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param metadata: The metadata for the image.
      :type metadata: dict, optional
      :param params: The parameters for the image normalization.
                     Defaults to NormalizeParams().
      :type params: NormalizeParams, optional
      :param \*\*kwargs: Additional keyword arguments.

      Defaults to NormalizeParams().

      :raises ValueError: Failed to normalize the image: {str(e)}

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: resize(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.convert_params.ResizeParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Resize the image data.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param metadata: The metadata for the image.
      :type metadata: dict, optional
      :param params: The parameters for the image resizing.
                     Defaults to ResizeParams().
      :type params: ResizeParams, optional
      :param \*\*kwargs: Additional keyword arguments.

      :raises ValueError: Failed to resize the image: {str(e)}

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: crop_images(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.convert_params.CropParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Crop the image data.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param metadata: The metadata for the image.
      :type metadata: dict, optional
      :param params: The parameters for the image cropping.
                     Defaults to CropParams().
      :type params: CropParams, optional
      :param \*\*kwargs: Additional keyword arguments.

      :raises ValueError: The crop size is larger than the image size.
      :raises ValueError: top_left must be provided when mode='topleft'.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


