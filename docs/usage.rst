.. currentmodule:: paidiverpy

Usage
=====

You can run your preprocessing pipeline using **Paidiverpy** in several ways, typically requiring just one to three lines of code:

1. **Python Package**: Install the package and utilize it in your Python scripts.

   .. code-block:: text

      In [1]: from paidiverpy.pipeline import Pipeline

      In [2]: pipeline = Pipeline(config_file_path="../examples/config_files/config_simple2.yaml")

      In [3]: pipeline.run()
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:24 | Running step 0: raw - OpenLayer
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:25 | Step 0 completed
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:25 | Running step 1: colour_correction - ColourLayer
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:25 | Step 1 completed


   In this example, we instantiate the `Pipeline` class and pass a configuration file containing the pipeline information and run the pipeline. The images will be processed as NumPy arrays.

   To view the pipeline details, simply print the pipeline object:

   .. code-block:: text

      In [5]: pipeline

   .. raw:: html
       :file: _static/pipeline.html

   To see a thumbnail of the output images, run the following code:

   .. code-block:: text

       In [6]: pipeline.images

   .. raw:: html
       :file: _static/pipeline_images.html

   To save the output images in the specified output directory from the configuration file, use the following command:

   .. code-block:: text

       In [7]: pipeline.save_images(image_format="png")

2. **Command Line Interface (CLI)**: Execute the package via the command line.

   You can run the package using the CLI with the following command:

   .. code-block:: bash

       paidiverpy -c "../examples/config_files/config_simple2.yaml"

3. **Docker**: Use the Docker image to run the package.

   After building or pushing the Docker image, you can run the package with the following command:

   .. code-block:: bash

       docker run --rm \
         -v <INPUT_PATH>:/app/input/ \
         -v <OUTPUT_PATH>:/app/output/ \
         -v <FULL_PATH_OF_CONFIGURATION_FILE_WITHOUT_FILENAME>:/app/config_files \
         paidiverpy \
         paidiverpy -c /app/examples/config_files/<CONFIGURATION_FILE_FILENAME>

   In this command:

   - `<INPUT_PATH>`: The input path defined in your configuration file, where the input images are located.
   - `<OUTPUT_PATH>`: The output path specified in your configuration file.
   - `<FULL_PATH_OF_CONFIGURATION_FILE_WITHOUT_FILENAME>`: The local directory containing your configuration file.
   - `<CONFIGURATION_FILE_FILENAME>`: The name of the configuration file.
