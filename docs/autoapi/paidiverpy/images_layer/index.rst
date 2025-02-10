paidiverpy.images_layer
=======================

.. py:module:: paidiverpy.images_layer

.. autoapi-nested-parse::

   Module to handle images and metadata for each step in the pipeline.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.images_layer.MAX_IMAGES_TO_SHOW
   paidiverpy.images_layer.NUM_CHANNELS_GRAY
   paidiverpy.images_layer.NUM_CHANNELS_RGBA
   paidiverpy.images_layer.NUM_DIMS_GRAY


Classes
-------

.. autoapisummary::

   paidiverpy.images_layer.ImagesLayer


Module Contents
---------------

.. py:data:: MAX_IMAGES_TO_SHOW
   :value: 12


.. py:data:: NUM_CHANNELS_GRAY
   :value: 1


.. py:data:: NUM_CHANNELS_RGBA
   :value: 4


.. py:data:: NUM_DIMS_GRAY
   :value: 2


.. py:class:: ImagesLayer(output_path: str | None = None)

   
   Class to handle images and metadata for each step in the pipeline.

   :param output_path: Path to save the images. Default is None.
   :type output_path: str















   ..
       !! processed by numpydoc !!

   .. py:attribute:: steps
      :value: []



   .. py:attribute:: step_metadata
      :value: []



   .. py:attribute:: images
      :value: []



   .. py:attribute:: max_images
      :value: 12



   .. py:attribute:: output_path


   .. py:attribute:: filenames
      :value: []



   .. py:method:: add_step(step: str, images: numpy.ndarray | dask.array.core.Array = None, metadata: pandas.DataFrame = None, step_metadata: dict | None = None, update_metadata: bool = False, track_changes: bool = True) -> None

      
      Add a step to the pipeline.

      :param step: The step to add
      :type step: str
      :param images: The images to add.
      :type images: Union[np.ndarray, da.core.Array], optional

      Defaults to None.
          metadata (pd.DataFrame, optional): The metadata to add. Defaults to None.
          step_metadata (dict, optional): The metadata for the step.
      Defaults to None.
          update_metadata (bool, optional): Whether to update the metadata.
          track_changes (bool, optional): Whether to track changes. Defaults to True.















      ..
          !! processed by numpydoc !!


   .. py:method:: remove_steps_by_name(step: tuple) -> int

      
      Remove steps by name.

      :param step: The step to remove
      :type step: str

      :returns: The index of the removed step
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: remove_steps_by_order(step_order: int) -> None

      
      Remove steps by order.

      :param step_order: The step order to remove
      :type step_order: int















      ..
          !! processed by numpydoc !!


   .. py:method:: get_last_step_order() -> int

      
      Get the last step order.

      :returns: The last step order
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: get_step(step: str | int | None = None, by_order: bool = False, last: bool = False) -> list[numpy.ndarray | dask.array.core.Array]

      
      Get a step by name or order.

      :param step: The step to get. Defaults to None.
      :type step: Union[str, int], optional
      :param by_order: If True, get the step by order. Defaults to False.
      :type by_order: bool, optional
      :param last: If True, get the last step. Defaults to False.
      :type last: bool, optional

      :returns: The images for the step
      :rtype: List[Union[np.ndarray, da.core.Array]]















      ..
          !! processed by numpydoc !!


   .. py:method:: show(image_number: int = 0) -> None

      
      Show the images in the pipeline.

      :param image_number: The index of the image to show. Defaults to 0.
      :type image_number: int, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: save(step: str | int | None = None, by_order: bool = False, last: bool = False, output_path: str | None = None, image_format: str = 'png', client: dask.distributed.Client = None, n_jobs: int = 1, logger: logging.Logger | None = None) -> None

      
      Save the images in the pipeline.

      :param step: The step to save. Defaults to None.
      :type step: Union[str, int], optional
      :param by_order: If True, save the step by order. Defaults to False.
      :type by_order: bool, optional
      :param last: If True, save the last step. Defaults to False.
      :type last: bool, optional
      :param output_path: The output path to save the images. Defaults to None.
      :type output_path: str, optional
      :param image_format: The image format to save. Defaults to "png".
      :type image_format: str, optional
      :param client: The Dask client. Defaults to None.
      :type client: Client, optional
      :param n_jobs: The number of jobs to use. Defaults to 1.
      :type n_jobs: int, optional
      :param logger: The logger to log messages. Defaults to None.
      :type logger: logging.Logger, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: save_remote(images: list[numpy.ndarray | dask.array.core.Array], output_path: str, image_format: str, client: dask.distributed.Client, n_jobs: int, step_order: int, logger: logging.Logger) -> None

      
      Save the images to a remote location.

      :param images: The images to save.
      :type images: list
      :param output_path: The output path to save the images.
      :type output_path: str
      :param image_format: The image format to save.
      :type image_format: str
      :param client: The Dask client.
      :type client: Client
      :param n_jobs: The number of jobs to use.
      :type n_jobs: int
      :param step_order: The step order.
      :type step_order: int
      :param logger: The logger to log messages.
      :type logger: logging.Logger















      ..
          !! processed by numpydoc !!


   .. py:method:: save_local(images: list[numpy.ndarray | dask.array.core.Array], output_path: str, image_format: str, client: dask.distributed.Client, n_jobs: int, step_order: int, logger: logging.Logger) -> None

      
      Save the images to a local location.

      :param images: The images to save.
      :type images: list
      :param output_path: The output path to save the images.
      :type output_path: str
      :param image_format: The image format to save.
      :type image_format: str
      :param client: The Dask client.
      :type client: Client
      :param n_jobs: The number of jobs to use.
      :type n_jobs: int
      :param step_order: The step order.
      :type step_order: int
      :param logger: The logger to log messages.
      :type logger: logging.Logger















      ..
          !! processed by numpydoc !!


   .. py:method:: process_and_upload(image: numpy.ndarray | dask.array.core.Array, img_path: str | pathlib.Path, image_format: str, s3_client: dask.distributed.Client | None = None) -> None

      
      Process and upload the images.

      :param image: The image to process and upload.
      :type image: Union[np.ndarray, da.core.Array]
      :param img_path: The image path to save.
      :type img_path: Union[str, Path]
      :param image_format: The image format to save.
      :type image_format: str
      :param s3_client: The S3 client. Defaults to None.
      :type s3_client: boto3.client, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_image(image: numpy.ndarray | dask.array.core.Array) -> tuple

      
      Calculate the image.

      :param image: The image to calculate.
      :type image: Union[np.ndarray, da.core.Array]

      :returns: The saved image and the colormap.
      :rtype: Tuple[np.ndarray, str]















      ..
          !! processed by numpydoc !!


   .. py:method:: remove(output_path: str | None = None) -> None

      
      Remove the images from the output path.

      :param output_path: The output path to save the images. Defaults to None.
      :type output_path: str, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: __repr__() -> str

      
      Return the string representation of the object.

      :returns: The string representation of the object
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: _repr_html_() -> str

      
      Return the HTML representation of the object.

      :returns: The HTML representation of the object
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: __call__(max_images: int | None = None) -> IPython.display.HTML

      
      Call the object.

      :param max_images: The maximum number of images to show.
      :type max_images: int, optional

      Defaults to None.

      :returns: The HTML representation of the object
      :rtype: HTML















      ..
          !! processed by numpydoc !!


   .. py:method:: _generate_html(max_images: int = 12, image_number: int | None = None) -> str

      
      Generate the HTML representation of the object.

      :param max_images: The maximum number of images to show. Defaults to 12.
      :type max_images: int
      :param image_number: The image number to show. Defaults to None.
      :type image_number: int, optional

      :returns: The HTML representation of the object
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: _generate_single_image_html(image_array: numpy.ndarray | dask.array.core.Array, step_index: int, image_index: int, size: tuple) -> str


   .. py:method:: numpy_array_to_base64(image_array: numpy.ndarray | dask.array.core.Array, size: tuple = (150, 150)) -> str
      :staticmethod:


      
      Convert a numpy array to a base64 image.

      :param image_array: The image array
      :type image_array: Union[np.ndarray, da.core.Array]
      :param size: _description_. Defaults to (150, 150).
      :type size: tuple, optional

      :returns: The base64 image
      :rtype: str















      ..
          !! processed by numpydoc !!


