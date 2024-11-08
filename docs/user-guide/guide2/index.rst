

Colour/content-based processing 
====================

Handling biodiversity images, especially those obtained in natural settings, presents numerous challenges. Issues such as inconsistent lighting, blurriness, and hazy veil over images can drastically compromise image quality and practical usability. This section is dedicated to the modification and enhancement of the visual characteristics of these images, aiming to enhance clarity, detail resolution, and interpretative accuracy. 

The challenges extend beyond natural environments, influencing diverse imaging scenarios, including underwater settings [SC, MI]. These challenges predominantly stem from: 

- **Light Attenuation and Environmental Interference**: Factors such as water turbidity or forest canopy cover can severely limit light, leading to poor contrast and obscured details.
  
- **Absorption and Scattering**: These processes, which respectively remove light energy and alter the direction of light paths, significantly degrade underwater image quality. Influenced by water temperature, salinity, and particles like marine snow, these factors affect the absorption coefficient, complicating the imaging process.

- **Camera and Sensor Limitations**: Variations in sensor sizes, sensitivities, and focal lengths affect the field of view and the overall imaging quality.
  
- **Artificial Lighting Drawbacks**: Utilized to counter low natural light, artificial lighting systems often suffer from non-uniform illumination, which can exacerbate scattering and absorption, sometimes creating overly bright spots in images.

Our preprocessing module is designed to tackle these challenges head-on, aiming to:

- **Correct Errors**: Rectify issues introduced by both camera hardware and environmental conditions during image capture.
  
- **Enhance Visual Clarity and Appeal**: Boost the visual quality of images for deeper analysis and interpretation, making it easier to discern fine details and subtle features.
  
- **Standardize Data**: Achieve consistency across images collected from different sources and under varied conditions, enhancing the reliability of data analyses and comparisons.

**Objective image quality metrics** can be categorized into three main types depending on the availability of a reference image:

- **Full reference metrics**: Requires an original, unaltered image for comparison.
- **Reduced-reference metrics**: Utilizes only partial information from the original image.
- **No-reference or "blind" quality assessment**: Does not rely on any reference image at all [SC].

In our case, where no original image is available for comparison, we must rely on no-reference metrics to quantitatively assess the effectiveness of our preprocessing techniques. Our focus here will be on the specific quantitative metrics that have been utilized by researchers to gauge the performance of preprocessing algorithms in such contexts.


1. Colour alteration
------------------

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

