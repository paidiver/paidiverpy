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

.. py:class:: ImagesLayer(output_path: str | pathlib.Path | None = None)

   
   Class to handle images and metadata for each step in the pipeline.

   :param output_path: Path to save the images. Default is None.
   :type output_path: str | Path | None















   ..
       !! processed by numpydoc !!

   .. py:method:: add_step(step: str, images: xarray.Dataset, step_metadata: dict[str, object], metadata: pandas.DataFrame | None = None, track_changes: bool = True) -> None

      
      Add a step to the pipeline.

      :param step: The step to add
      :type step: str
      :param images: The images for the step.
      :type images: xr.Dataset
      :param step_metadata: The metadata for the step.
      :type step_metadata: dict
      :param metadata: The metadata to set. Defaults to None.
      :type metadata: pd.DataFrame | None, optional
      :param track_changes: Whether to track changes. Defaults to True.
      :type track_changes: bool, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: replace_step(images: xarray.Dataset) -> None

      
      Add a step to the pipeline.

      :param images: The images for the step.
      :type images: xr.Dataset















      ..
          !! processed by numpydoc !!


   .. py:method:: set_images(images: xarray.Dataset) -> None

      
      Set the images for the layer.

      :param images: The images to set.
      :type images: xr.Dataset















      ..
          !! processed by numpydoc !!


   .. py:method:: remove_steps_by_order(step_order: int) -> None

      
      Remove steps by order.

      :param step_order: The step order to remove
      :type step_order: int















      ..
          !! processed by numpydoc !!


   .. py:method:: get_step(step: str | int | None = None, last: bool = False, flag: None | int = None) -> xarray.Dataset

      
      Get a step by name or order.

      :param step: The step to get. Defaults to None.
      :type step: str | int, optional
      :param last: If True, get the last step. Defaults to False.
      :type last: bool, optional
      :param flag: The flag to filter the images. Defaults to None.
      :type flag: None | int, optional

      :returns: The images for the step or None if the step does not exist.
      :rtype: xr.Dataset | None















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


   .. py:method:: save(config: paidiverpy.config.configuration.Configuration, step: str | int | None = None, last: bool = True, output_path: str | pathlib.Path | None = None, image_format: str = 'png', client: dask.distributed.Client | None = None, n_jobs: int = 1, use_dask: bool = False) -> None

      
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
      :param client: The Dask client. Defaults to None.
      :type client: Client, optional
      :param n_jobs: The number of jobs to use. Defaults to 1.
      :type n_jobs: int, optional
      :param use_dask: Whether to use Dask. Defaults to False.
      :type use_dask: bool, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: process_and_upload(image: numpy.ndarray[Any, Any] | dask.array.core.Array, img_path: str | pathlib.Path, image_format: str, s3_client: dask.distributed.Client | None = None) -> None

      
      Process and upload the images.

      :param image: The image to process and upload.
      :type image: np.ndarray | da.core.Array
      :param img_path: The image path to save.
      :type img_path: str | Path
      :param image_format: The image format to save.
      :type image_format: str
      :param s3_client: The S3 client. Defaults to None.
      :type s3_client: boto3.client, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_image(image: numpy.ndarray[Any, Any] | dask.array.core.Array) -> numpy.ndarray[Any, Any]

      
      Calculate the image.

      :param image: The image to calculate.
      :type image: np.ndarray | da.core.Array

      :returns: The calculated image.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: remove(output_path: str | pathlib.Path | None = None) -> None

      
      Remove the images from the output path.

      :param output_path: The output path to save the images. Defaults to None.
      :type output_path: str | Path, optional















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


   .. py:method:: process_single_image(img: numpy.ndarray[Any, Any], height: numpy.ndarray[Any, Any], width: numpy.ndarray[Any, Any], filename: numpy.ndarray[Any, Any], output_path: pathlib.Path, image_format: str, s3_client: dask.distributed.Client | None, processor: collections.abc.Callable) -> int
      :staticmethod:


      
      Process a single image and save it.

      :param img: The image to process.
      :type img: np.ndarray
      :param height: The height of the image.
      :type height: np.ndarray
      :param width: The width of the image.
      :type width: np.ndarray
      :param filename: The filename of the image.
      :type filename: np.ndarray
      :param output_path: The path to save the output.
      :type output_path: Path
      :param image_format: The format to save the image.
      :type image_format: str
      :param s3_client: The S3 client to use for uploading.
      :type s3_client: Client | None
      :param processor: The processing function to use.
      :type processor: Callable

      :returns: The status code (0 for success).
      :rtype: int















      ..
          !! processed by numpydoc !!


