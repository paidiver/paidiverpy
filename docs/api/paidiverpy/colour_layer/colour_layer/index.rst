paidiverpy.colour_layer.colour_layer
====================================

.. py:module:: paidiverpy.colour_layer.colour_layer

.. autoapi-nested-parse::

   Colour layer module.

   This module contains the ColourLayer class for processing the images in the
   colour layer.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.colour_layer.colour_layer.ColourLayer


Module Contents
---------------

.. py:class:: ColourLayer(parameters: dict, config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   ColourLayer class.

   This class contains the methods for processing the images in the colour layer.

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

   .. py:method:: grayscale(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.GrayScaleParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Convert the image to grayscale.

      Method to convert the image to grayscale.

      :param image_data: The input image.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Parameters for the grayscale conversion.
                     Defaults to GrayScaleParams().
      :type params: GrayScaleParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: If the input image does not have 3 channels or 4 channels with alpha.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: gaussian_blur(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.GaussianBlurParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Gaussian blur.

      Method to apply Gaussian blur to the image.

      :param image_data: The image to apply Gaussian blur.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: the parameters for the method.
                     Defaults to GaussianBlurParams().
      :type params: GaussianBlurParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: Error applying Gaussian blur.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: sharpen(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.SharpenParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Sharpening.

      Method to apply sharpening to the image.

      :param image_data: The image to apply sharpening.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Params for method. Defaults to SharpenParams().
      :type params: SharpenParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: Error applying sharpening.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: contrast_adjustment(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.ContrastAdjustmentParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Contrast adjustment.

      Method to apply contrast adjustment to the image.

      :param image_data: The image to apply contrast adjustment.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Params for method.
                     Defaults to ContrastAdjustmentParams().
      :type params: ContrastAdjustmentParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: Error applying contrast adjustment.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: illumination_correction(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.IlluminationCorrectionParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Illumination correction.

      Method to apply illumination correction to the image.

      :param image_data: The image to apply illumination correction.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Params for method.
                     Defaults to IlluminationCorrectionParams().
      :type params: IlluminationCorrectionParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: Error applying illumination correction.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: deblur(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.DeblurParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Deblurring.

      Method to apply deblurring to the image.

      :param image_data: The image to apply deblurring.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Params for method.
                     Defaults to DeblurParams().
      :type params: DeblurParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: Unknown PSF type. Please use 'gaussian' or 'motion'.
      :raises ValueError: Unknown method type. Please use 'wiener'.
      :raises NotImplementedError: Unknown method type. Please use 'wiener'.
      :raises ValueError: Error applying contrast adjustment.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: colour_alteration(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.ColourAlterationParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Apply colour alteration to the image.

      :param image_data: The image to alter colour channel.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Params for method. Defaults to None.
      :type params: ColourAlterationParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises ValueError: Unknown method type. Please use 'white_balance'.
      :raises ValueError: Image is gray-scale'.
      :raises e: Error applying colour alteration.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: edge_detection(image_data: numpy.ndarray, metadata: dict | None = None, params: paidiverpy.models.colour_params.EdgeDetectionParams = None, **kwargs: dict) -> tuple[numpy.ndarray, dict]
      :staticmethod:


      
      Edge detection.

      Method to apply edge detection to the image.

      :param image_data: The image to apply edge detection.
      :type image_data: np.ndarray
      :param metadata: Metadata for the image.
      :type metadata: dict, optional
      :param params: Params for method.
                     Defaults to EdgeDetectionParams().
      :type params: EdgeDetectionParams, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :raises e: Error applying edge detection.

      :returns: The updated image and the updated metadata.
      :rtype: tuple[np.ndarray, dict]















      ..
          !! processed by numpydoc !!


   .. py:method:: get_object_features(gray_image_data: numpy.ndarray, label_image_data: numpy.ndarray, params: paidiverpy.models.colour_params.EdgeDetectionParams) -> tuple[dict, numpy.ndarray]
      :staticmethod:


      
      Get object features.

      Get the features of the object.

      :param gray_image_data: The grayscale image data.
      :type gray_image_data: np.ndarray
      :param label_image_data: The label image data.
      :type label_image_data: np.ndarray
      :param params: The parameters for edge detection.
      :type params: EdgeDetectionParams

      :returns: The features of the object and the binary image data.
      :rtype: tuple[dict, np.ndarray]















      ..
          !! processed by numpydoc !!


   .. py:method:: gaussian_psf(size: list[int], sigma: float) -> numpy.ndarray
      :staticmethod:


      
      Gaussian point spread function.

      Create a Gaussian point spread function (PSF).

      :param size: The size of the PSF.
      :type size: List[int]
      :param sigma: The standard deviation of the PSF.
      :type sigma: float

      :returns: The Gaussian PSF.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: motion_psf(size: list[float], length: float, angle_xy: float, angle_z: int = 0) -> numpy.ndarray
      :staticmethod:


      
      Motion point spread function.

      Create a motion point spread function (PSF).

      :param size: size of the PSF
      :type size: float[]
      :param length: length of the PSF
      :type length: float
      :param angle_xy: angle of the PSF
      :type angle_xy: float
      :param angle_z: tilt in the z-axis. Defaults to 0.
      :type angle_z: int, optional

      :returns: The motion PSF
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: deconvolution(img: numpy.ndarray, bw_img: numpy.ndarray, blurd_bw_img: numpy.ndarray, deconv: bool, deconv_method: str, deconv_iter: int, deconv_mask_weight: float, small_float_val: float = 1e-06) -> numpy.ndarray
      :staticmethod:


      
      Deconvolution.

      Perform deconvolution on the image.

      :param img: The image to deconvolve
      :type img: np.ndarray
      :param bw_img: The binary image to use for deconvolution
      :type bw_img: np.ndarray
      :param blurd_bw_img: The blurred binary image to use for deconvolution
      :type blurd_bw_img: np.ndarray
      :param deconv: Whether to perform deconvolution
      :type deconv: bool
      :param deconv_method: The method to use for deconvolution
      :type deconv_method: str
      :param deconv_iter: The number of iterations for deconvolution
      :type deconv_iter: int
      :param deconv_mask_weight: The weight for the deconvolution mask
      :type deconv_mask_weight: float
      :param small_float_val: The small float value. Defaults to 1e-6.
      :type small_float_val: float, optional

      :returns: The deconvolved image
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: sharpness_analysis(gray_img: numpy.ndarray, img: numpy.ndarray, features: dict, estimate_sharpness: bool = True) -> dict
      :staticmethod:


      
      Sharpness analysis.

      Estimate the sharpness of the image using FFTs.

      :param gray_img: The grayscale image
      :type gray_img: np.ndarray
      :param img: The image
      :type img: np.ndarray
      :param features: The features of the image
      :type features: dict
      :param estimate_sharpness: Whether to estimate sharpness.
      :type estimate_sharpness: bool, optional

      Defaults to True.

      :returns: The features of the image
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: detect_edges(img: numpy.ndarray, method: str, blur_radius: float, threshold: dict) -> numpy.ndarray
      :staticmethod:


      
      Detect edges.

      Detect edges in the image.

      :param img: The image to detect edges
      :type img: np.ndarray
      :param method: The method to use for edge detection
      :type method: str
      :param blur_radius: The radius for the blur
      :type blur_radius: float
      :param threshold: The threshold for edge detection
      :type threshold: dict

      :returns: The filled edges
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: process_edges(edges_mag: numpy.ndarray, low_threshold: float, blur_radius: float) -> numpy.ndarray
      :staticmethod:


      
      Process the edges.

      Process the edges using the low threshold.

      :param edges_mag: The edges magnitude
      :type edges_mag: np.ndarray
      :param low_threshold: The low threshold
      :type low_threshold: float
      :param blur_radius: The radius for the blur
      :type blur_radius: float

      :returns: The filled edges
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: process_edges_mean(edges_mag: numpy.ndarray, blur_radius: float) -> numpy.ndarray
      :staticmethod:


      
      Process the edges.

      Process the edges using the mean.

      :param edges_mag: The edges magnitude
      :type edges_mag: np.ndarray
      :param blur_radius: The radius for the blur
      :type blur_radius: float

      :returns: The filled edges
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: make_gaussian(size: int, fwhm: int = 3, center: tuple | None = None) -> numpy.ndarray
      :staticmethod:


      
      Make a square gaussian kernel.

      Method to make a square gaussian kernel.

      :param size: The size of the square.
      :type size: int
      :param fwhm: The full-width-half-maximum. Defaults to 3.
      :type fwhm: int, optional
      :param center: The center of the square. Defaults to None.
      :type center: tuple, optional

      :returns: The square gaussian kernel.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


