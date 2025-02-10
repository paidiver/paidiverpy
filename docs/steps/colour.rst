.. _step_colour:

Colour Layer
============

The Colour Layer is responsible for applying colour correction and enhancements to images. It supports various modes, allowing users to perform grayscale conversion, blurring, edge detection, sharpening, contrast adjustment, deblurring, illumination correction, colour alteration, etc. The Colour Layer supports the following modes:

- **Grayscale (grayscale)**: Converts the image to grayscale.

  - ``keep_alpha`` (bool, default=False): Whether to preserve the alpha channel.
  - ``method`` (str, default="opencv"): The grayscale conversion method.
  - ``invert_colours`` (bool, default=False): If True, inverts the grayscale values.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Gaussian Blur (gaussian_blur)**: Applies a Gaussian blur to smooth the image.

  - ``sigma`` (float, default=1.0): The standard deviation for the Gaussian kernel.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Edge Detection (edge_detection)**: Detects edges in the image using various methods.

  - ``method`` (str, default="sobel"): The edge detection method.
  - ``blur_radius`` (float, default=1.0): The radius for pre-blurring the image.
  - ``threshold`` (float, default=0.1): The threshold for edge detection.
  - ``object_type`` (str, default="bright"): The type of object to detect ("bright" or "dark").
  - ``object_selection`` (str, default="largest"): The selection criteria for detected objects.
  - ``estimate_sharpness`` (bool, default=False): Whether to estimate image sharpness.
  - ``deconv`` (bool, default=False): Whether to apply deconvolution.
  - ``deconv_method`` (str, default="LR"): The deconvolution method.
  - ``deconv_iter`` (int, default=10): The number of iterations for deconvolution.
  - ``deconv_mask_weight`` (float, default=0.03): The weight for the deconvolution mask.
  - ``small_float_val`` (float, default=1e-6): A small float value for numerical stability.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Sharpening (sharpen)**: Enhances the sharpness of the image.

  - ``alpha`` (float, default=1.5): The weight of the original image.
  - ``beta`` (float, default=-0.5): The weight of the sharpening filter.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Contrast Adjustment (contrast)**: Adjusts the contrast of the image.

  - ``method`` (str, default="clahe"): The contrast adjustment method.
  - ``kernel_size`` (int, optional): The size of the CLAHE kernel.
  - ``clip_limit`` (float, default=0.01): The clip limit for CLAHE.
  - ``gamma_value`` (float, default=0.5): The gamma correction value.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Deblurring (deblur)**: Attempts to restore sharpness to a blurred image.

  - ``method`` (str, default="wiener"): The deblurring method.
  - ``psf_type`` (str, default="gaussian"): The type of point spread function (PSF).
  - ``sigma`` (float, default=20): The standard deviation for Gaussian PSF.
  - ``angle`` (int, default=45): The motion blur angle.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Illumination Correction (illumination_correction)**: Corrects illumination variations in the image.

  - ``method`` (str, default="rolling"): The illumination correction method.
  - ``radius`` (int, default=100): The rolling radius for background estimation.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

- **Colour Alteration (colour_alteration)**: Modifies the colour balance of the image.

  - ``method`` (str, default="white_balance"): The colour alteration method.
  - ``raise_error`` (bool, default=False): Whether to raise an error on failure.

Each mode provides a configurable set of parameters, allowing fine-grained control over the applied transformations.
