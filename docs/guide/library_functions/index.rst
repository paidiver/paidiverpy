.. _guide_library_functions:

Using OpenCV and scikit-image functions
========================================

The ``library`` layer lets a pipeline call public functions from OpenCV and
scikit-image without writing a custom Paidiverpy class. Each configured
function is invoked for every image and must return an image as a NumPy array.

OpenCV example
--------------

Create a configuration such as ``config_library.yml``:

.. code-block:: yaml

   general:
     sample_data: plankton_csv
     output_path: ./output
     n_jobs: 1

   steps:
     - library:
         name: opencv_blur
         library: opencv
         function: GaussianBlur
         params:
           ksize:
             $tuple: [5, 5]
           sigmaX: 1.0

Run it using the normal pipeline API:

.. code-block:: python

   from paidiverpy.pipeline import Pipeline

   pipeline = Pipeline(config_file_path="config_library.yml")
   pipeline.run()

scikit-image example
--------------------

Function paths are relative to ``skimage``:

.. code-block:: yaml

   steps:
     - library:
         name: skimage_blur
         library: skimage
         function: filters.gaussian
         params:
           sigma: 1.5
           channel_axis: -1
           preserve_range: true

Special YAML values
-------------------

YAML lists remain Python lists. Use ``$tuple`` when an API requires a tuple:

.. code-block:: yaml

   ksize:
     $tuple: [5, 5]

Use ``$ref`` when a parameter requires an imported constant, dtype, or public
callable:

.. code-block:: yaml

   borderType:
     $ref: cv2.BORDER_REFLECT

References may start with ``cv2``, ``skimage``, ``numpy``, or ``np``. Private
symbols and other packages are rejected.

Selecting an image result
-------------------------

Some functions return more than one value. ``cv2.threshold``, for example,
returns the calculated threshold followed by the output image. Select the
image using ``result_index``:

.. code-block:: yaml

   - library:
       name: threshold
       library: opencv
       function: threshold
       result_index: 1
       params:
         thresh: 127
         maxval: 255
         type:
           $ref: cv2.THRESH_BINARY

When the image is not the first argument, set ``image_param`` to its keyword
name. The same key must not also appear under ``params``.

Limitations
-----------

The layer is for image-to-image functions. It rejects scalar results, empty
arrays, object arrays, and arrays with dimensions other than two or three.
Functions producing measurements, tables, feature collections, or a different
output size for every image are not compatible with this layer. OpenCV colour
functions may use BGR ordering, while Paidiverpy images are normally RGB; add
an explicit colour conversion when the selected algorithm depends on channel
order. scikit-image may return floating-point arrays, and the returned dtype
and value range are preserved.

The runnable notebook is available at
``examples/example_notebooks/library_layer_tutorial.ipynb``.
