.. currentmodule:: paidiverpy

Usage
=====

You can run your preprocessing pipeline using **Paidiverpy** in several ways, typically requiring just one to three lines of code:

1. **Python Package**: Install the package and utilize it in your Python scripts.

   .. code-block:: text

      In [1]: from paidiverpy.pipeline import Pipeline

      In [2]: pipeline = Pipeline(config_file_path="../examples/config_files/config_simple2.yml")

      In [3]: pipeline.run()
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:24 | Running step 0: raw - OpenLayer
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:25 | Step 0 completed
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:25 | Running step 1: colour_correction - ColourLayer
      ☁ paidiverpy ☁  |       INFO | 2024-11-04 17:49:25 | Step 1 completed


   In this example, we instantiate the `Pipeline` class and pass a configuration file containing the pipeline information and run the pipeline.

   For more details on the configuration file format, refer to the :doc:`configuration_file` section.
   The images will be processed as NumPy arrays.

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

       paidiverpy -c "../examples/config_files/config_simple2.yml"

3. **Docker**: Use the Docker image to run the package.

   After building or pushing the Docker image, you can run the package with the following command:

   .. code-block:: bash

        docker run --rm \
          -v <INPUT_PATH>:/app/input/ \
          -v <OUTPUT_PATH>:/app/output/ \
          -v <METADATA_PATH>:/app/metadata/ \
          -v <CONFIG_DIR>:/app/config_files/ \
          paidiverpy -c /app/examples/config_files/<CONFIG_FILE>

   In this command:

   - `<INPUT_PATH>`: Local directory containing input images (as defined in the configuration file).
   - `<OUTPUT_PATH>`: Local directory where processed images will be saved.
   - `<METADATA_PATH>`: Local directory containing the metadata file.
   - `<CONFIG_DIR>`: Local directory containing the configuration file.
   - `<CONFIG_FILE>`: Name of the configuration file.

   If you are using remote data from an object store, it is not necessary to create volumes for the remote data. However, if you want to upload images to an object store, you need to pass an environment file in the `docker run` command, as shown below:

   .. code-block:: bash

        docker run --rm \
          -v <CONFIG_DIR>:/app/config_files/ \
          --env-file .env \
          paidiverpy -c /app/examples/config_files/<CONFIG_FILE>

   In this case, you have to create a `.env` file with your object store credentials:

   .. code-block:: text

        OS_SECRET=your_secret
        OS_TOKEN=your_token
        OS_ENDPOINT=your_endpoint

4. **GUI**: Use the graphical user interface to run the package.

   You need to have the `panel` package installed to use the GUI. If you don't have it installed, you can do so with the following command:

    .. code-block:: text

        pip install panel

   With the package installed, you can run the package using the GUI by executing the following command:

   .. code-block:: bash

       paidiverpy -gui


   The GUI will open in your default web browser, where you can select your configuration file and run the pipeline interactively.
   More details on the GUI can be found in the :ref:`gui` section.
