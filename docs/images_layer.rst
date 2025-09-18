.. _images_layer:

How we handle the images
========================

In the Paidiverpy package, images are managed through the :class:`ImagesLayer` class.
This class is the central container for storing and tracking the results of each image
processing step in a pipeline. Every operation performed on images produces a new
*step* that is added to the layer, together with associated metadata.

Why Xarray?
-----------

Images in Paidiverpy are stored as :class:`xarray.Dataset` objects instead of plain
NumPy arrays. This allows us to:

- Keep images and their metadata tightly coupled.
- Track changes across multiple processing steps without losing information.
- Handle both sequential and parallel (Dask-backed) image processing seamlessly.
- Easy to visualize and export images.
- Easy to implement complex operations like broadcasting and merging.

Consistent Dimensions with Padding
----------------------------------

Unlike plain arrays, all images in an :class:`ImagesLayer` **must share the same
dimensions** (`x`, `y`, and `band`) so that they can coexist inside a single
`xarray.Dataset`. However, real-world images often have different sizes.

To solve this, Paidiverpy applies the following strategy:

1. **Padding** – All images are padded to the largest `height` and `width` in the batch.
2. **Tracking original size** – Each image stores its true
   :code:`original_height` and :code:`original_width` as coordinates.
3. **Processing** – Before running any operation, the image is cropped back to
   its original size.
4. **Re-padding** – After processing, the result is written back into a padded array,
   so the dataset remains consistent across all images.

This ensures that processing functions always work on the *true content* of the
image, while the dataset structure remains compatible with xarray’s broadcasting
and merging rules.

Dataset Structure Across Steps
------------------------------

Each processing step creates a new set of variables in the dataset.

- The processed image itself is stored under a variable named:
  ``images_0``, ``images_1``, ``images_2`` … one for each step.

- Each step also has its **own dimensions and coordinates**, suffixed with the
  step number, for example:
  ``x_0``, ``y_0``, ``band_0``, ``original_height_0``, ``original_width_0`` are unique to the first step.

- The only coordinate shared across all steps is **``filename``**, which serves as
  the index for all images.

This means that variables and dimensions are always step-specific, and the
``_number`` suffix ties them to the corresponding step.

Accessing Steps
---------------

Each processing step is stored under unique variable names
(`images_0`, `images_1`, etc.) along with its own dimensions and coordinates.

The :func:`get_step` method allows extracting any step (by index or name) and
automatically restores the **standard dimension names** (`x`, `y`, `band`,
`original_height`, `original_width`) so that the output looks like a normal
image dataset again.

---

This structure makes it possible to build reproducible pipelines where
images, metadata, and processing history remain fully synchronized.
