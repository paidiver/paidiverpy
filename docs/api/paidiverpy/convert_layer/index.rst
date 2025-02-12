paidiverpy.convert_layer
========================

.. py:module:: paidiverpy.convert_layer

.. autoapi-nested-parse::

   
   __init__.py file for convert_layer module.
















   ..
       !! processed by numpydoc !!


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/paidiverpy/convert_layer/convert_layer/index


Classes
-------

.. autoapisummary::

   paidiverpy.convert_layer.ConvertLayer


Package Contents
----------------

.. py:class:: ConvertLayer(config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.config.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, parameters: dict | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

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

   .. py:method:: convert_bits(image_data: numpy.ndarray, params: paidiverpy.config.convert_params.BitParams = None) -> numpy.ndarray
      :staticmethod:


      
      Convert the image to the specified number of bits.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters for the bit conversion.
      :type params: BitParams, optional

      Defaults to BitParams().

      :returns: The image data with the specified number of bits.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: channel_convert(image_data: numpy.ndarray, params: paidiverpy.config.convert_params.ToParams = None) -> numpy.ndarray
      :staticmethod:


      
      Convert the image to the specified channel.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters for the channel conversion.
      :type params: ToParams, optional

      Defaults to ToParams().

      :raises ValueError: The image is already in RGB format.
      :raises ValueError: The image is already in grayscale.
      :raises ValueError: Failed to convert the image to {params.to}: {str(e)}

      :returns: The image data with the specified channel.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_bayer_pattern(image_data: numpy.ndarray, params: paidiverpy.config.convert_params.BayerPatternParams = None) -> numpy.ndarray
      :staticmethod:


      
      Convert the image to the specified Bayer pattern.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters for the Bayer pattern conversion.
      :type params: BayerPatternParams, optional

      Defaults to BayerPatternParams().

      :raises ValueError: Invalid Bayer pattern for a single-channel image.
      :raises KeyError: Invalid Bayer pattern for a single-channel image.

      Expected 'RG', 'BG', 'GR', or 'GB'.

      :returns: The image data with the specified Bayer pattern.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: normalize_image(image_data: numpy.ndarray, params: paidiverpy.config.convert_params.NormalizeParams = None) -> numpy.ndarray
      :staticmethod:


      
      Normalize the image data.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters for the image normalization.
      :type params: NormalizeParams, optional

      Defaults to NormalizeParams().

      :raises ValueError: Failed to normalize the image: {str(e)}

      :returns: The normalized image data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: resize(image_data: numpy.ndarray, params: paidiverpy.config.convert_params.ResizeParams = None) -> numpy.ndarray
      :staticmethod:


      
      Resize the image data.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters for the image resizing.
      :type params: ResizeParams, optional

      Defaults to ResizeParams().

      :raises ValueError: Failed to resize the image: {str(e)}

      :returns: The resized image data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: crop_images(image_data: numpy.ndarray, params: paidiverpy.config.convert_params.CropParams = None) -> numpy.ndarray
      :staticmethod:


      
      Crop the image data.

      :param image_data: The image data.
      :type image_data: np.ndarray
      :param params: The parameters for the image cropping.
      :type params: CropParams, optional

      Defaults to CropParams().

      :raises ValueError: Crop range is out of bounds.
      :raises ValueError: Failed to crop the image: {str(e)}

      :returns: The cropped image data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


