paidiverpy.open_layer.utils
===========================

.. py:module:: paidiverpy.open_layer.utils

.. autoapi-nested-parse::

   Open Layer utils module.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.open_layer.utils.open_image_remote
   paidiverpy.open_layer.utils.open_image_local
   paidiverpy.open_layer.utils.correct_image_dims_and_format
   paidiverpy.open_layer.utils.pad_image
   paidiverpy.open_layer.utils.load_raw_image
   paidiverpy.open_layer.utils.load_raw_image_using_path_open
   paidiverpy.open_layer.utils.decode_8bpp
   paidiverpy.open_layer.utils.decode_16bpp
   paidiverpy.open_layer.utils.extract_exif_single


Module Contents
---------------

.. py:function:: open_image_remote(img_path: str, image_type: str, image_open_args: dict[str, Any], **kwargs: dict[str, Any]) -> tuple[numpy.ndarray[Any, Any] | dask.array.core.Array, dict[str, Any], str]

   
   Open an image file.

   :param img_path: The path to the image file
   :type img_path: str
   :param image_type: The image type
   :type image_type: str
   :param image_open_args: The image open arguments
   :type image_open_args: dict[str, Any]
   :param \*\*kwargs: Additional keyword arguments. The following are supported:
                      - storage_options (dict[str, Any]): The storage options for reading metadata file.
   :type \*\*kwargs: dict[str, Any]

   :raises ValueError: Failed to open the image

   :returns: The image data, the EXIF data, and the filename
   :rtype: tuple[np.ndarray[Any, Any] | da.core.Array, dict[str, Any], str]















   ..
       !! processed by numpydoc !!

.. py:function:: open_image_local(img_path: str, image_type: str, image_open_args: dict[str, Any], **_kwargs: dict[str, Any]) -> tuple[numpy.ndarray[Any, Any] | dask.array.core.Array, dict[str, Any], str]

   
   Open an image file.

   :param img_path: The path to the image file
   :type img_path: str
   :param image_type: The image type
   :type image_type: str
   :param image_open_args: The image open arguments
   :type image_open_args: dict
   :param \*\*_kwargs: Additional keyword arguments. This is just a place holder for the code
   :type \*\*_kwargs: dict

   :raises ValueError: Failed to open the image

   :returns: The image data, the EXIF data, and the filename
   :rtype: tuple[np.ndarray[Any, Any] | da.core.Array, dict, str]















   ..
       !! processed by numpydoc !!

.. py:function:: correct_image_dims_and_format(img: numpy.ndarray[Any, Any] | dask.array.core.Array, image_type: str | None = None) -> numpy.ndarray[Any, Any] | dask.array.core.Array

   
   Correct the image dimensions and format.

   :param img: The image data
   :type img: np.ndarray[Any, Any] | da.core.Array
   :param image_type: The image type
   :type image_type: str | None

   :returns: The corrected image data
   :rtype: np.ndarray[Any, Any] | da.core.Array















   ..
       !! processed by numpydoc !!

.. py:function:: pad_image(img: numpy.ndarray[Any, Any] | dask.array.core.Array, target_height: int, target_width: int) -> numpy.ndarray[Any, Any] | dask.array.core.Array

   
   Pad the image to the target height and width.

   :param img: The image data
   :type img: np.ndarray[Any, Any] | da.core.Array
   :param target_height: The target height
   :type target_height: int
   :param target_width: The target width
   :type target_width: int

   :returns: The padded image
   :rtype: np.ndarray[Any, Any] | da.core.Array















   ..
       !! processed by numpydoc !!

.. py:function:: load_raw_image(img_path: str | io.BytesIO, image_type: str, image_open_args: dict[str, Any], remote: bool = False) -> numpy.ndarray[Any, Any] | dask.array.core.Array

   
   Load a raw image file.

   :param img_path: The path to the image file or a BytesIO object
   :type img_path: str | BytesIO
   :param image_type: The image type
   :type image_type: str | None
   :param image_open_args: The image open arguments
   :type image_open_args: dict | None
   :param remote: Whether the image is remote or local. Defaults to False.
   :type remote: bool

   :raises ValueError: Failed to open the image

   :returns: The loaded image data
   :rtype: np.ndarray[Any, Any]















   ..
       !! processed by numpydoc !!

.. py:function:: load_raw_image_using_path_open(img_path: str | io.BytesIO, image_open_args: dict[str, Any], remote: bool = False) -> numpy.ndarray[Any, Any] | dask.array.core.Array

   
   Load a raw image file using the open function.

   :param img_path: The path to the image file or a BytesIO object
   :type img_path: str
   :param image_open_args: The image open arguments
   :type image_open_args: dict | None
   :param remote: Whether the image is remote or local. Defaults to False.
   :type remote: bool

   :raises ValueError: Failed to open the image

   :returns: The loaded image data
   :rtype: np.ndarray[Any, Any]















   ..
       !! processed by numpydoc !!

.. py:function:: decode_8bpp(img: numpy.ndarray[Any, Any], image_misc: list[str], width: int, height: int, channels: int, bayer_pattern: str | None = None) -> numpy.ndarray[Any, Any]

   
   Decode 8-bit per channel image data.

   :param img: The image data.
   :type img: np.ndarray[Any, Any]
   :param image_misc: The image metadata.
   :type image_misc: list[str]
   :param width: The width of the image.
   :type width: int
   :param height: The height of the image.
   :type height: int
   :param channels: The number of channels in the image.
   :type channels: int
   :param bayer_pattern: The Bayer pattern if the image is in Bayer format. Defaults to None.
   :type bayer_pattern: str | None

   :returns: The decoded image data.
   :rtype: np.ndarray[Any, Any]















   ..
       !! processed by numpydoc !!

.. py:function:: decode_16bpp(img: numpy.ndarray[Any, Any], layout: str = '5:6:5', width: int = 2448, height: int = 2048, endianess: str | None = None) -> numpy.ndarray[Any, Any]

   
   Decode 16-bit packed RGB into 8-bit per channel RGB based on layout.

   :param img: The packed 16-bit image data.
   :type img: np.ndarray[Any, Any]
   :param layout: The layout of the packed data. Valid options include "5:6:5", "5:5:5", "5:5:6".
   :type layout: str
   :param width: The width of the image.
   :type width: int
   :param height: The height of the image.
   :type height: int
   :param endianess: Whether to swap the byte order.
   :type endianess: bool

   :returns: The unpacked 8-bit RGB image data.
   :rtype: np.ndarray[Any, Any]















   ..
       !! processed by numpydoc !!

.. py:function:: extract_exif_single(img_path: str | io.BytesIO, image_type: str, image_name: str | None = None) -> dict[str, Any]

   
   Extract EXIF data from a single image file.

   :param img_path: The path to the image file or a BytesIO object.
   :type img_path: str | BytesIO
   :param image_type: The image type.
   :type image_type: str
   :param image_name: The name of the image file. Defaults to None.
   :type image_name: str, optional

   :returns: The EXIF data.
   :rtype: dict















   ..
       !! processed by numpydoc !!

