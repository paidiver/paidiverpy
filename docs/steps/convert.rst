.. _step_convert:

Convert Layer
=============

The convert layer is responsible for transforming images by changing their bit depth, format, size, and other properties. It provides several methods for these transformations, each with specific parameters. The Convert Layer supports the following modes:

- **Bit Conversion (bits)**: Converts the bit depth of an image.

  - `output_bits` (int): Target bit depth (default: 8).
  - `raise_error` (bool): Whether to raise an error if conversion fails (default: False).

- **Channel Conversion (to)**: Converts the image to a different data type.

  - `to` (str): Target data type (default: "uint8").
  - `channel_selector` (int): Selected channel index (default: 0).
  - `raise_error` (bool): Whether to raise an error if conversion fails (default: False).

- **Normalization (normalize)**: Normalizes pixel values to a specified range.

  - `min` (float): Minimum value for normalization (default: 0).
  - `max` (float): Maximum value for normalization (default: 1).
  - `raise_error` (bool): Whether to raise an error if normalization fails (default: False).

- **Resizing (resize)**: Resizes an image to specified dimensions.

  - `min` (int): Minimum size for resizing (default: 256).
  - `max` (int): Maximum size for resizing (default: 256).
  - `raise_error` (bool): Whether to raise an error if resizing fails (default: False).

- **Cropping (crop)**: Crops an image to a specified region.

  - `x` (tuple): Crop range along the x-axis (default: (0, -1)).
  - `y` (tuple): Crop range along the y-axis (default: (0, -1)).
  - `raise_error` (bool): Whether to raise an error if cropping fails (default: False).

Each mode provides a configurable set of parameters, allowing fine-grained control over the applied transformations.
