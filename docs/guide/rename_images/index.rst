.. _guide_rename_images:

Rename Images
=============

The `rename_images` method in **paidiverpy** provides a way to rename image files based on a specified strategy. This feature helps standardise filenames, ensuring consistency across datasets.

Configuration
-------------

To enable automatic renaming, set the `general.rename` parameter in the configuration file to one of the supported strategies.

.. code-block:: yaml

    general:
      rename: "datetime"
      image_open_args: "JPG"


The package currently supports two renaming strategies:

1. **Datetime-based (`datetime`)**
   - Filenames are generated from the image timestamp in the format:

     .. code-block:: text

         YYYYMMDDTHHMMSS.sssZ

   - If duplicate filenames occur, a suffix `_NUMBER` is added.

2. **UUID-based (`UUID`)**
   - Each filename is replaced with a randomly generated UUID.
