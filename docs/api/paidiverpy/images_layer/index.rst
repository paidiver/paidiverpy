paidiverpy.images_layer
=======================

.. py:module:: paidiverpy.images_layer

.. autoapi-nested-parse::

   Module to handle images and metadata for each step in the pipeline.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.images_layer.ImagesLayer


Module Contents
---------------

.. py:class:: ImagesLayer(output_path: str | None = None)

   
   Class to handle images and metadata for each step in the pipeline.

   :param output_path: Path to save the images. Default is None.
   :type output_path: str















   ..
       !! processed by numpydoc !!

   .. py:method:: add_step(step: str, images: numpy.ndarray | dask.array.core.Array = None, metadata: pandas.DataFrame = None, step_metadata: dict | None = None, update_metadata: bool = False, track_changes: bool = True) -> None

      
      Add a step to the pipeline.

      :param step: The step to add
      :type step: str
      :param images: The images to add.
      :type images: np.ndarray | da.core.Array, optional

      Defaults to None.
          metadata (pd.DataFrame, optional): The metadata to add. Defaults to None.
          step_metadata (dict, optional): The metadata for the step.
      Defaults to None.
          update_metadata (bool, optional): Whether to update the metadata.
          track_changes (bool, optional): Whether to track changes. Defaults to True.















      ..
          !! processed by numpydoc !!


   .. py:method:: remove_steps_by_order(step_order: int) -> None

      
      Remove steps by order.

      :param step_order: The step order to remove
      :type step_order: int















      ..
          !! processed by numpydoc !!


   .. py:method:: get_step(step: str | int | None = None, last: bool = False) -> list[numpy.ndarray | dask.array.core.Array]

      
      Get a step by name or order.

      :param step: The step to get. Defaults to None.
      :type step: str | int, optional
      :param last: If True, get the last step. Defaults to False.
      :type last: bool, optional

      :returns: The images for the step
      :rtype: list[np.ndarray | da.core.Array]















      ..
          !! processed by numpydoc !!


   .. py:method:: show(image_number: int = 0) -> IPython.display.HTML

      
      Show the images in the pipeline.

      :param image_number: The index of the image to show. Defaults to 0.
      :type image_number: int, optional

      :returns: The HTML representation of the images
      :rtype: HTML















      ..
          !! processed by numpydoc !!


   .. py:method:: save(step: str | int | None = None, last: bool = True, output_path: str | None = None, image_format: str = 'png', config: paidiverpy.config.configuration.Configuration | None = None, metadata: pandas.DataFrame | None = None, client: dask.distributed.Client = None, n_jobs: int = 1, logger: logging.Logger | None = None) -> None

      
      Save the images in the pipeline.

      :param step: The step to save. Defaults to None.
      :type step: str| int, optional
      :param last: If True, save the last step. Defaults to False.
      :type last: bool, optional
      :param output_path: The output path to save the images. Defaults to None.
      :type output_path: str, optional
      :param image_format: The image format to save. Defaults to "png".
      :type image_format: str, optional
      :param config: The configuration object. Defaults to None.
      :type config: Configuration, optional
      :param metadata: The metadata object. Defaults to None.
      :type metadata: pd.DataFrame, optional
      :param client: The Dask client. Defaults to None.
      :type client: Client, optional
      :param n_jobs: The number of jobs to use. Defaults to 1.
      :type n_jobs: int, optional
      :param logger: The logger to log messages. Defaults to None.
      :type logger: logging.Logger, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: save_remote(images: list[numpy.ndarray | dask.array.core.Array], output_path: str, image_format: str, metadata: pandas.DataFrame, client: dask.distributed.Client, n_jobs: int, step_order: int, logger: logging.Logger) -> None

      
      Save the images to a remote location.

      :param images: The images to save.
      :type images: list
      :param output_path: The output path to save the images.
      :type output_path: str
      :param image_format: The image format to save.
      :type image_format: str
      :param metadata: The metadata object.
      :type metadata: pd.DataFrame
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


   .. py:method:: save_local(images: list[numpy.ndarray | dask.array.core.Array], output_path: str, image_format: str, metadata: pandas.DataFrame, client: dask.distributed.Client, n_jobs: int, step_order: int, logger: logging.Logger) -> None

      
      Save the images to a local location.

      :param images: The images to save.
      :type images: list
      :param output_path: The output path to save the images.
      :type output_path: str
      :param image_format: The image format to save.
      :type image_format: str
      :param metadata: The metadata object.
      :type metadata: pd.DataFrame
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


   .. py:method:: process_and_upload(image: numpy.ndarray | dask.array.core.Array, img_path: str | pathlib.Path, image_format: str, s3_client: dask.distributed.Client | None = None, metadata: pandas.DataFrame | None = None) -> None

      
      Process and upload the images.

      :param image: The image to process and upload.
      :type image: np.ndarray | da.core.Array
      :param img_path: The image path to save.
      :type img_path: str | Path
      :param image_format: The image format to save.
      :type image_format: str
      :param s3_client: The S3 client. Defaults to None.
      :type s3_client: boto3.client, optional
      :param metadata: The metadata object. Defaults to None.
      :type metadata: pd.DataFrame, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_image(image: numpy.ndarray | dask.array.core.Array) -> tuple

      
      Calculate the image.

      :param image: The image to calculate.
      :type image: np.ndarray | da.core.Array

      :returns: The saved image and the colormap.
      :rtype: tuple[np.ndarray, str]















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


   .. py:method:: __call__(max_images: int | None = None) -> IPython.display.HTML

      
      Call the object.

      :param max_images: The maximum number of images to show.
      :type max_images: int, optional

      Defaults to None.

      :returns: The HTML representation of the object
      :rtype: HTML















      ..
          !! processed by numpydoc !!


