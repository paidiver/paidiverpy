paidiverpy.library_layer
========================

.. py:module:: paidiverpy.library_layer

.. autoapi-nested-parse::

   
   Generic OpenCV and scikit-image processing layer.
















   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.library_layer.LibraryLayer


Package Contents
----------------

.. py:class:: LibraryLayer(parameters: dict[str, Any], config_params: dict[str, Any] | paidiverpy.config.config_params.ConfigParams | None = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration | None = None, metadata: paidiverpy.metadata_parser.MetadataParser | None = None, images: paidiverpy.images_layer.ImagesLayer | None = None, paidiverpy: paidiverpy.Paidiverpy | None = None, step_name: str | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Apply a configured OpenCV or scikit-image function to every image.
















   ..
       !! processed by numpydoc !!

   .. py:method:: run() -> None

      
      Resolve the configured function and apply it to every image.
















      ..
          !! processed by numpydoc !!


   .. py:method:: resolve_function(library: str, path: str) -> collections.abc.Callable[..., Any]
      :staticmethod:


      
      Resolve a public callable below an approved library namespace.

      :param library: The library name, e.g., "opencv" or "skimage".
      :type library: str
      :param path: The dotted path to the function, e.g., "imgproc.cvtColor".
      :type path: str

      :returns: The resolved function.
      :rtype: Callable[..., Any]















      ..
          !! processed by numpydoc !!


   .. py:method:: decode_value(value: Any) -> Any
      :staticmethod:


      
      Recursively translate ``$ref`` and ``$tuple`` YAML representations.

      :param value: The value to decode.
      :type value: Any

      :returns: The decoded value.
      :rtype: Any















      ..
          !! processed by numpydoc !!


   .. py:method:: validate_result(result: Any) -> numpy.ndarray[Any, Any]
      :staticmethod:


      
      Ensure that a library call produced an image compatible with Paidiverpy.

      :param result: The result to validate.
      :type result: Any

      :returns: The validated image array.
      :rtype: np.ndarray[Any, Any]















      ..
          !! processed by numpydoc !!


