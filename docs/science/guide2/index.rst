.. _science_guide2:

Colour/content-based processing
===============================

Introduction
------------

Handling biodiversity images, especially those obtained in natural settings, presents numerous challenges. Issues such as inconsistent lighting, blurriness, and a hazy veil over images can drastically compromise image quality and practical usability. This section is dedicated to the modification and enhancement of the visual characteristics of these images, aiming to enhance clarity, detail resolution, and interpretative accuracy.

The challenges extend beyond natural environments, influencing diverse imaging scenarios, including underwater settings [SC, MI]. These challenges predominantly stem from:

- **Light Attenuation and Environmental Interference**: Factors such as water turbidity or forest canopy cover can severely limit light, leading to poor contrast and obscured details.
- **Absorption and Scattering**: These processes, which respectively remove light energy and alter the direction of light paths, significantly degrade underwater image quality. Influenced by water temperature, salinity, and particles like marine snow, these factors affect the absorption coefficient, complicating the imaging process.
- **Camera and Sensor Limitations**: Variations in sensor sizes, sensitivities, and focal lengths affect the field of view and the overall imaging quality.
- **Artificial Lighting Drawbacks**: Utilized to counter low natural light, artificial lighting systems often suffer from non-uniform illumination, which can exacerbate scattering and absorption, sometimes creating overly bright spots in images.

Our preprocessing module is designed to tackle these challenges head-on, aiming to:

- **Correct Errors**: Rectify issues introduced by both camera hardware and environmental conditions during image capture.
- **Enhance Visual Clarity and Appeal**: Boost the visual quality of images for deeper analysis and interpretation, making it easier to discern fine details and subtle features.
- **Standardize Data**: Achieve consistency across images collected from different sources and under varied conditions, enhancing the reliability of data analyses and comparisons.

Objective image quality metrics can be categorized into three main types depending on the availability of a reference image:

- **Full reference metrics**: Requires an original, unaltered image for comparison.
- **Reduced-reference metrics**: Utilizes only partial information from the original image.
- **No-reference or "blind" quality assessment**: Does not rely on any reference image at all [SC].

In our case, where no original image is available for comparison, we must rely on no-reference metrics to quantitatively assess the effectiveness of our preprocessing techniques. Our focus here will be on the specific quantitative metrics that have been utilized by researchers to gauge the performance of preprocessing algorithms in such contexts.


[SC] Schettini, R., Corchs, S. "Underwater image processing: state of the art of restoration and image enhancement methods." EURASIP Journal on Advances in Signal Processing, 2010.
[MI] Massot-Campos, M., Yamada, T., Thornton, B. "Towards sensor agnostic artificial intelligence for underwater imagery." IEEE Underwater Technology, 2023.

---

1. Colour alteration
--------------------

**Overview**

Colour alteration is essential for correcting colour distortions in underwater imagery. As depth increases, colours diminish at different rates based on their wavelengths. Red is the first to disappear at approximately 3 meters, followed by orange at 5 meters, yellow at 10 meters, and eventually green and purple at greater depths. Blue, due to its shorter wavelength, travels the furthest, resulting in underwater images being dominated by blue-green hues [RA]. Additionally, variations in light sources contribute to non-uniform colour casts, which are characteristic of typical underwater images. The primary challenge is light absorption, which is wavelength-dependent, causing a progressive loss of colours with increasing depth [SA].

**Methodology**

- **Local Histogram Equalization** [GA]: Enhances colour contrast locally within different regions of the image.
- **Automatic Colour Equalization (White balance adjustment)** [CH]: Adjusts the colours automatically based on statistical analysis of the image's colour distribution.
- **Local adaptive contrast enhancement** [ZH]
- **Contrast-limited histogram equalization** [SA]

**Challenges**

Some challenges arise during the colour alteration process:

- **Subjectivity**: The "best" settings for colour alterations can be subjective and dependent on the intended use of the images.

**Success Metrics**

The success of colour alteration is measured by comparing processed images with true colour charts or known references. Accurate colour correction should result in images that closely match the expected real-world colours of the objects and environments depicted.


[RA] Schettini, R., Corchs, S. "Underwater Image Processing: State of the Art of Restoration and Image Enhancement Methods." EURASIP Journal on Advances in Signal Processing, 2010.
[SA] Sankpal, S.S., Deshpande, S.S. "A review on image enhancement and color correction techniques for underwater images." Advances in Computational Sciences and Technology, 2016.
[GA] Garcia, R., Nicosevici, T., Cufí, X. "On the way to solve lighting problems in underwater imaging." IEEE, 2002.
[CH] Chambah, M., Semani, D., Renouf, A., et al. "Underwater color constancy: enhancement of automatic live fish recognition." SPIE, 2003.
[ZH] Zhang, W., Zhuang, P., Sun, H.H., et al. "Underwater image enhancement via minimal color loss and locally adaptive contrast enhancement." IEEE Transactions on Image Processing, 2022.

---

2. Contrast alteration
----------------------

**Overview**

Contrast alteration is essential for enhancing images, especially those taken underwater. This module modifies the intensity of dark and bright areas in an image, greatly improving visual clarity. Capturing high-resolution underwater images is difficult due to issues like low contrast, poor visibility, and blurriness. Light attenuation, caused mainly by absorption and scattering, reduces image quality. Scattering is categorized into forward scattering, which causes blurriness, and backward scattering, which adds noise [AI]. Additionally, marine snow—particles suspended in the water—further degrades image quality [SC].

**Methodology**

Contrast alteration involves adjusting the histogram of pixel intensity values in an image, a technique known as histogram equalization. This method enhances the distinction between features by modifying the light and dark areas. Advanced approaches include:

- **Adaptive Histogram Equalization (AHE)**: Enhances contrast by analyzing localized intensity distributions, improving visibility without increasing noise.
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Refines AHE by applying contrast limiting, avoiding over-enhancement in specific blocks, for balanced enhancement throughout the image.
- **Gamma Correction**: A non-linear transformation to adjust brightness and contrast.

**Challenges**

Several challenges complicate the process of contrast alteration:

- **Judgment**: The optimal level of enhancement is subjective and depends on the image's intended use and personal preference.
- **Poor Quality**: Low initial quality can limit enhancement effectiveness.
- **Over-Enhancement**: Excessive adjustment can create unnatural appearances and detail loss.

**Success Metrics**

The success of contrast alteration is measured by the clarity of image details and improved feature visibility, essential for tasks such as species identification.


[AI] Almutiry, O., et al. "Underwater images contrast enhancement and its challenges: a survey." Multimedia Tools and Applications, 2024.
[SC] Schettini, R., Corchs, S. "Underwater image processing: state of the art of restoration and image enhancement methods." EURASIP Journal on Advances in Signal Processing, 2010.

---

3. Backscatter removal
----------------------

**Overview**

Backscatter is a common issue in underwater photography, caused by light reflecting off particles suspended in water or air. This scattered light results in a hazy appearance and reduced visibility, creating a foggy effect that increases with distance. Backscatter leads to a significant loss of contrast in images, and its removal can enhance clarity and contrast [TS].

**Methodology**

Backscatter removal is typically achieved using dehazing algorithms, involving:

- **Estimation of Backscatter**: A low-pass filter estimates spatial distribution of optical backscatter.
- **Subtraction and Scaling**: The estimated backscatter is subtracted, followed by scaling to restore contrast.

**Challenges**

- **Automation**: Fully automating backscatter removal remains challenging.
- **Detail Preservation**: Excessive removal can eliminate essential image details.

**Success Metrics**

Success is measured by increased contrast and reduced haze, ensuring essential details are preserved.


[TS] Tsiotsios, C., et al. "Backscatter compensated photometric stereo with 3 sources." IEEE, 2014.

---

4. Illumination correction
--------------------------

**Overview**

Illumination correction addresses uneven lighting, ensuring uniform distribution. In underwater imaging, this often involves lens vignetting and light source positioning challenges. Vignetting darkens edges due to lens design, and autonomous vehicle lighting geometry introduces bright and dark zones, affecting colour and image quality.

**Methodology**

- **Background Subtraction**: Estimates and subtracts background illumination.
- **Histogram Matching**: Normalizes light distribution by matching image histograms to reference values.

**Challenges**

- **Dynamic Lighting**: Consistent results are difficult with variable lighting.
- **Information Loss**: Over-correction risks detail loss.

**Success Metrics**

Effective illumination correction minimizes over/underexposed areas, providing a clear view.


[ST] Sternberg, S.R. "Biomedical image processing." Computer, 1983.

---

5. De-blurring
--------------

**Overview**

Blurring in images can result from camera shake, focus issues, or subject movement, and it significantly degrades image quality and obscures important details. Blurring is influenced by various factors, including the imaging environment and the stability of the camera during image capture. To address blurring, especially without prior knowledge of the blur's cause, blind image deblurring techniques are employed [ZHA]. The degradation model for a blurry image can be expressed as:

.. math::
   S = H * U + N

where :math:`S` is the blurry image, :math:`H` is the blur kernel (point spread function, PSF), :math:`U` is the original clear image, and :math:`N` represents noise. The circular PSF, characterized by its radius :math:`R`, is a common approximation for out-of-focus distortion.

**Methodology**

Restoring a blurred image involves estimating the original image from the degraded version using several techniques:

- **Wiener Filter**: A widely used method for de-blurring, the Wiener filter restores the image by reducing the effect of noise and blur, assuming certain known parameters such as the signal-to-noise ratio (SNR) and the blur kernel's characteristics.
- **Blind Deconvolution**: An iterative method that estimates both the original image and the blur kernel, often used when the blur characteristics are unknown.
- **Total Variation (TV) Regularization**: Reduces noise and preserves edges by applying regularization based on total variation of the image intensity.

**Challenges**

Several challenges complicate the de-blurring process:

- **Noise Trade-off**: While de-blurring can sharpen images, it may also increase noise, especially in low-light conditions, leading to a potential trade-off between sharpness and noise levels.
- **Unknown Blur Kernel**: If the blur characteristics are unknown, it is difficult to fully restore the original image, necessitating adaptive or blind deblurring techniques.

**Success Metrics**

The success of de-blurring is measured by the sharpness and clarity of the restored image. Sharper images with more defined details indicate successful de-blurring. However, it is essential to balance the enhancement of image details with the potential increase in noise, ensuring that the final image remains useful for analysis and interpretation.


[ZHA] Zhang, Z., Zheng, L., Piao, Y., Tao, S., Xu, W., Gao, T., and Wu, X. "Blind remote sensing image deblurring using local binary pattern prior." Remote Sensing, 2022.
