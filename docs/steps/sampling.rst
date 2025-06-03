.. _step_sampling:

Sampling Layer
==============

The resample layer is responsible for resampling data based on various criteria such as datetime, depth, altitude, and other parameters. The resampling process helps in standardizing data for analysis. The resample layer supports the following modes:

- **Datetime Resampling (datetime)**: Samplings data within a specified datetime range.

  - ``min`` (str, optional): Minimum datetime value.
  - ``max`` (str, optional): Maximum datetime value.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Depth Resampling (depth)**: Samplings data based on depth.

  - ``by`` (str, default="lower"): Strategy for depth selection.
  - ``value`` (float, optional): Depth value for resampling.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Altitude Resampling (altitude)**: Samplings data based on altitude.

  - ``value`` (float, optional): Altitude value for resampling.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Pitch and Roll Resampling (pitch_roll)**: Samplings data based on pitch and roll values.

  - ``pitch`` (float, optional): Pitch value.
  - ``roll`` (float, optional): Roll value.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Overlapping Resampling (overlapping)**: Samplings data based on overlapping conditions.

  - ``omega`` (float, default=0.5): Overlapping factor.
  - ``theta`` (float, default=0.5): Angle threshold.
  - ``threshold`` (float, optional): Threshold for resampling.
  - ``camera_distance`` (float, default=1.12): Camera distance for calculations.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Fixed Number Resampling (fixed)**: Samplings data to a fixed number of points.

  - ``value`` (int, default=10): Number of points to resample to.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Percent Resampling (percent)**: Samplings data based on a percentage of the dataset.

  - ``value`` (float, default=0.1): Percentage of data to keep.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Region-Based Resampling (region)**: Samplings data based on specified regions.

  - ``file`` (str, optional): File containing region definitions.
  - ``limits`` (list[str], optional): List of region limits.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Obscure Photos Resampling (obscure)**: Samplings data based on obscure photo detection.

  - ``min`` (int, default=0): Minimum threshold for obscurity.
  - ``max`` (int, default=1): Maximum threshold for obscurity.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

Each mode provides a configurable set of parameters, allowing fine-grained control over the applied transformations.
