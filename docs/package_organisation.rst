.. _package_organisation:

Package Organisation
====================

The Paidiverpy package is structured into multiple layers, each designed to facilitate specific aspects of image processing. This modular approach allows for flexibility, ease of use, and clarity in managing different functionalities.

.. image:: _static/paidiver_organisation.jpg

Core Classes
------------------

- **Paidiverpy**: The main class that serves as the central container for all image processing functions. It orchestrates the execution of various processing tasks and provides a unified interface for users. This class is responsible for initializing and managing the entire pipeline, making it easier to perform complex operations with minimal setup.

- **OpenLayer**: A subclass dedicated to handling the opening and initial loading of images. This layer is responsible for reading images from specified input paths and preparing them for subsequent processing steps. It supports various image formats, ensuring compatibility with the input data. For more information, see the :ref:`configuration_file` section.

- **ConvertLayer**: This layer focuses on image format conversion and bit-depth adjustments. Users can apply various conversion techniques, such as changing colour spaces or altering image bit depth, to prepare images for specific analytical tasks. For more information, see the :ref:`step_convert` section.

- **PositionLayer**: Handles spatial information and metadata associated with images. This layer allows users to apply transformations based on the geographic coordinates or other spatial parameters, making it suitable for tasks that require spatial awareness. For more information, see the :ref:`step_position` section.

- **SamplingLayer**: Responsible for resampling images to different resolutions or formats. This layer ensures that images can be resized or interpolated as needed, providing the necessary flexibility for different analysis scenarios. For more information, see the :ref:`step_sampling` section.

- **ColourLayer**: This layer applies colour manipulation techniques to images, such as colour correction, filtering, and enhancement. It offers a range of methods to adjust the visual properties of images, allowing users to improve their quality for better analysis. For more information, see the :ref:`step_colour` section.

- **CustomLayer**: A versatile layer that enables users to define custom processing functions. This layer provides a high degree of flexibility, allowing users to implement specific algorithms or processing steps tailored to their needs. For more information, see the :ref:`step_custom` section.

Supporting Classes
------------------

- **Configuration**: A utility class that parses and manages configuration files. It ensures that the pipeline runs according to user specifications by loading and validating the settings defined in the configuration files. This class is essential for maintaining reproducibility in processing workflows. For more information, see the :ref:`configuration_file` section.

- **Metadata**: Manages metadata associated with images. This class is responsible for loading, validating, and processing metadata files, ensuring that the required contextual information is readily available for image processing tasks. It supports both IFDO and CSV formats, making it versatile for different use cases. For more information, see the :ref:`images_metadata` section.

- **ImagesLayer**: This class stores the outputs from each image processing step. It acts as a container for processed images, allowing users to access intermediate results easily. This organization helps in tracking changes made to images throughout the processing pipeline.

- **InvestigationLayer**: This layer is designed for exploratory data analysis and visualization. It provides tools for generating visual representations of images and their associated metadata, helping users gain insights into the data. For more information, see the :ref:`guide_test_mode` section.

Pipeline Class
------------------

The **Pipeline** class integrates all processing steps defined in the configuration file into a cohesive workflow. It orchestrates the execution of each layer in the correct sequence, ensuring that the input data flows smoothly through the processing stages. Users can customize the pipeline by modifying the configuration file, allowing for tailored processing that meets specific analytical needs.
