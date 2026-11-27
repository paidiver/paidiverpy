.. _step_library:

Library Layer
=============

The library layer applies a public OpenCV or scikit-image function to every
image in a pipeline. Images are passed as NumPy arrays and the function must
return a non-empty, two- or three-dimensional NumPy array. A failure on any
image stops the pipeline.

The available fields are:

``library``
  Required. Either ``opencv`` or ``skimage``.

``function``
  Required. The public dotted path below the selected library, such as
  ``GaussianBlur`` or ``filters.gaussian``.

``params``
  Optional keyword arguments passed to the function. Ordinary YAML values are
  passed directly. ``$tuple`` creates a Python tuple and ``$ref`` resolves a
  public value from ``cv2``, ``skimage``, ``numpy``, or ``np``.

``image_param``
  Optional keyword receiving the current image. When omitted, the image is
  passed as the first positional argument.

``result_index``
  Optional index selecting the image from a tuple or list result. It is useful
  for functions such as ``cv2.threshold``.

Example
-------

.. code-block:: yaml

   steps:
     - library:
         name: gaussian_blur
         library: opencv
         function: GaussianBlur
         params:
           ksize:
             $tuple: [5, 5]
           sigmaX: 1.5
           borderType:
             $ref: cv2.BORDER_REFLECT

See :ref:`guide_library_functions` for a complete tutorial and more examples.
