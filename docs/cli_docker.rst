************************
CLI and Docker Reference
************************

This page documents how to run Paidiverpy from the command line and through
Docker. It is based on the current CLI implementation in the package.


Overview
--------

Paidiverpy exposes a single command-line entrypoint:

.. code-block:: bash

      paidiverpy [OPTIONS]

The CLI supports four main workflows:

1. Run an image-processing pipeline from a YAML configuration file.
2. Validate a configuration file.
3. Run benchmark tests.
4. Launch the GUI (Panel app).


CLI Arguments
-------------

You can view all options with:

.. code-block:: bash

      paidiverpy --help

Supported arguments:

- ``-c``, ``--configuration_file`` (string)
     Path to a configuration file. Default: ``./config/config.yml``.

- ``-v``, ``--validate`` (flag)
     Validate the configuration file and exit.

- ``-bt``, ``--benchmark_test`` (JSON string)
     Run benchmark tests using a JSON payload.

- ``-gui``, ``--gui`` (optional arguments)
     Launch the Paidiverpy GUI.


Run a Pipeline
--------------

The most common usage is to run a pipeline from a config file:

.. code-block:: bash

      paidiverpy -c examples/config_files/config_simple.yml

Behavior:

- Loads and executes the configured processing steps.
- Saves images according to the output settings in the configuration.


Validate a Configuration
------------------------

To validate a config file without running the full pipeline:

.. code-block:: bash

      paidiverpy -c examples/config_files/config_simple.yml --validate

This is useful for CI checks and quick local verification.


Run Benchmark Tests
-------------------

Benchmark mode accepts a JSON string with benchmark parameters.
Example:

.. code-block:: bash

      paidiverpy \
           -c examples/config_files/config_benchmark.yml \
           -bt '{"cluster_type":"local","cores":[1,2,4],"processes":[1,2,4],"memory":[1,2,4],"scale":[1,2]}'

Expected keys typically include:

- ``cluster_type``
- ``cores``
- ``processes``
- ``memory``
- ``scale``

Tip: quote the full JSON string to avoid shell parsing issues.


Launch the GUI
--------------

GUI mode starts a Panel server for interactive execution.

Basic usage:

.. code-block:: bash

      paidiverpy -gui

Custom Panel arguments can also be passed, for example:

.. code-block:: bash

      paidiverpy -gui "--port 5007 --address 127.0.0.1"

Notes:

- The ``panel`` executable must be available in your environment.
- Default GUI server parameters are:

     - ``--port 5006``
     - ``--address 0.0.0.0``
     - ``--autoreload``


Docker Reference
----------------

Paidiverpy Docker images use ``paidiverpy`` as the container entrypoint, so
arguments can be passed directly after the image name.

Build locally:

.. code-block:: bash

      docker build -t paidiverpy -f dockerfiles/Dockerfile .

Or pull from GHCR:

.. code-block:: bash

      docker pull ghcr.io/paidiver/paidiverpy:latest
      docker tag ghcr.io/paidiver/paidiverpy:latest paidiverpy:latest


Run Pipeline in Docker
----------------------

For local datasets, mount the paths used by your configuration file:

.. code-block:: bash

      docker run --rm \
           -v <INPUT_PATH>:/app/input/ \
           -v <OUTPUT_PATH>:/app/output/ \
           -v <METADATA_PATH>:/app/metadata/ \
           -v <CONFIG_DIR>:/app/config_files/ \
           paidiverpy -c /app/config_files/<CONFIG_FILE>

Where:

- ``<INPUT_PATH>`` is your host directory with input images.
- ``<OUTPUT_PATH>`` is your host output directory.
- ``<METADATA_PATH>`` is your host directory for metadata files.
- ``<CONFIG_DIR>`` is your host directory containing the YAML config.
- ``<CONFIG_FILE>`` is the config file name.

Important path behavior in Docker:

- When running inside Docker, the CLI uses the basename of the config file and
     resolves it under ``/app/config_files``.
- In practice, mount your config directory to ``/app/config_files`` and pass
     either:

     - ``-c /app/config_files/<CONFIG_FILE>``
     - or ``-c <CONFIG_FILE>``


Remote Object Storage with Docker
---------------------------------

If your input data is read directly from object storage, you may not need local
volume mounts for input data. If you upload results to object storage, provide
credentials through an environment file.

Example:

.. code-block:: bash

      docker run --rm \
           -v <CONFIG_DIR>:/app/config_files/ \
           --env-file .env \
           paidiverpy -c /app/config_files/<CONFIG_FILE>

Example ``.env`` file:

.. code-block:: text

      OS_SECRET=your_secret
      OS_TOKEN=your_token
      OS_ENDPOINT=your_endpoint


Run GUI in Docker
-----------------

To run the GUI in a container and access it from your browser:

.. code-block:: bash

      docker run --rm -p 5006:5006 paidiverpy -gui

If you use custom GUI options, map the same port in ``-p`` and in the ``-gui``
arguments.


Troubleshooting
---------------

- If ``paidiverpy`` is not found:
     ensure the package is installed and your environment is activated.

- If GUI fails with a message about ``panel``:
     install Panel in the active environment (``pip install panel``).

- If Docker cannot find your configuration:
     verify the config directory is mounted to ``/app/config_files`` and that the
     config filename is correct.

- If input/output paths fail in Docker:
     ensure your YAML paths match mounted container paths (for example,
     ``/app/input`` and ``/app/output``).
