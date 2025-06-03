paidiverpy.utils.formating_html
===============================

.. py:module:: paidiverpy.utils.formating_html

.. autoapi-nested-parse::

   HTML output utilities for paidiverpy.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.formating_html.metadata_repr
   paidiverpy.utils.formating_html.pipeline_repr
   paidiverpy.utils.formating_html.config_repr
   paidiverpy.utils.formating_html.images_repr
   paidiverpy.utils.formating_html.generate_single_image_html
   paidiverpy.utils.formating_html.numpy_array_to_base64


Module Contents
---------------

.. py:function:: metadata_repr(metadata: paidiverpy.metadata_parser.MetadataParser) -> str

   
   Represents metadata as an HTML string.

   :param metadata: The metadata object to represent.
   :type metadata: MetadataParser

   :returns: String representation of the metadata.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: pipeline_repr(pipeline: paidiverpy.Paidiverpy) -> str

   
   Generate HTML representation of the pipeline.

   :param pipeline: The pipeline instance to represent.
   :type pipeline: Paidiverpy

   :returns: The HTML representation of the pipeline.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: config_repr(config: paidiverpy.config.configuration.Configuration) -> str

   
   Generate HTML representation of the config.

   :param config: The configuration instance to represent.
   :type config: Configuration

   :returns: The HTML representation of the config.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: images_repr(images: paidiverpy.images_layer.ImagesLayer, max_images: int = 12, image_number: int | None = None, html: bool = False) -> str

   
   Generate the HTML representation of the object.

   :param images: The ImagesLayer object to represent.
   :type images: ImagesLayer
   :param max_images: The maximum number of images to show. Defaults to 12.
   :type max_images: int
   :param image_number: The image number to show. Defaults to None.
   :type image_number: int, optional
   :param html: If True, the output will be in HTML format. Defaults to False.
   :type html: bool

   :returns: The HTML representation of the object
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: generate_single_image_html(image_array: numpy.ndarray | dask.array.core.Array, filenames: list[str], step_index: int, image_index: int, size: tuple, random_id: str) -> str

   
   Generate HTML for a single image.

   :param image_array: The image array
   :type image_array: np.ndarray | da.core.Array
   :param filenames: The filenames of the images
   :type filenames: list[str]
   :param step_index: The index of the step
   :type step_index: int
   :param image_index: The index of the image
   :type image_index: int
   :param size: The size of the image
   :type size: tuple
   :param random_id: The random id for the image
   :type random_id: str

   :returns: The HTML for the image
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: numpy_array_to_base64(image_array: numpy.ndarray | dask.array.core.Array, size: tuple = (150, 150)) -> str

   
   Convert a numpy array to a base64 image.

   :param image_array: The image array
   :type image_array: np.ndarray | da.core.Array
   :param size: _description_. Defaults to (150, 150).
   :type size: tuple, optional

   :returns: The base64 image
   :rtype: str















   ..
       !! processed by numpydoc !!

