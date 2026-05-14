.. _benchmarks:

Benchmarks
==========

This section summarises the benchmark tests used to evaluate the performance of
**Paidiverpy**.

The benchmarks measure execution time, memory usage, and scalability across
different execution configurations. The aim is to help users understand how
Paidiverpy performs under different resource settings and to provide practical
guidance for tuning pipelines on local machines and HPC systems.

Paidiverpy also provides a benchmark utility that users can run with their own
datasets and pipeline configurations. The examples at the end of this section
show how to launch custom benchmark tests from the command line.


About the tests
---------------

The benchmark experiments use the **benthic_ifdo** dataset, described in the
:ref:`example data <example_data>` section. This dataset contains benthic images
and metadata from the Haig Fras area in the UK, collected in 2012.

The full dataset is publicly available through the British Oceanographic Data
Centre (BODC): `BODC published dataset <https://www.bodc.ac.uk/data/published_data_library/catalogue/10.5285/093edbc7-3552-3d35-e063-6c86abc099d5/>`_

For the benchmark tests, we use a subset of approximately 290 MB containing 100
images with varying resolutions.

The preprocessing pipeline used for the benchmark is defined in the following
YAML configuration file:

.. literalinclude:: ../src/paidiverpy/utils/benchmark/config_benchmark.yml
   :language: yaml

This configuration applies several preprocessing operations, mainly from the
**ColourLayer** components. The benchmark utility runs this
pipeline multiple times while varying the execution settings, such as the number
of jobs, workers, threads, and available memory, depending on the selected
execution mode.


Test configurations
-------------------

The benchmarks were run using the following execution modes.

**Serial execution**

Serial execution runs the pipeline with a single job. This provides a baseline
for comparing parallel configurations.

.. code-block:: yaml

   n_jobs: [1]

**Thread-based execution**

Thread-based execution varies only the ``n_jobs`` parameter. This mode is useful
for testing parallel execution on a single machine without creating a Dask
cluster.

.. code-block:: yaml

   n_jobs: [4, 8, 16]

**Slurm execution**

The Slurm benchmark was run on an HPC system using a batch allocation with
64 GB of memory and 32 CPU cores. The benchmark varied ``n_jobs`` to measure how
the pipeline scales within the allocated resources.

.. code-block:: yaml

   n_jobs: [1, 2, 4, 8, 16, 32]


Results
-------

The detailed benchmark results are stored in JSON format and are available here:

`benchmark.json </_static/benchmark.json>`_

This file contains the execution time and memory usage recorded for each tested
configuration.

A graphical summary of the benchmark results is shown below:

.. raw:: html

   <iframe src="_static/benchmark.html"
           width="100%"
           height="650px"
           style="border:none;">
   </iframe>

The results show that execution time decreases as the number of parallel jobs
increases, demonstrating that Paidiverpy can benefit from parallel execution.

The Slurm configuration provides the strongest performance improvements in this
benchmark, particularly at higher values of ``n_jobs``. This indicates that
Paidiverpy can make effective use of larger HPC allocations for image
preprocessing workloads.

.. admonition:: Note

   Serial execution took more than 2 hours to complete. It is therefore not
   included in the plot, so that the differences between the parallel
   configurations are easier to compare.


How to run your own benchmarks
------------------------------

You can run benchmark tests with your own datasets and pipeline configurations
using the Paidiverpy CLI.

1. **Prepare a dataset**

   Ensure that your dataset is supported by Paidiverpy and that the input data
   and metadata are accessible from the machine or cluster where the benchmark
   will run.

2. **Create a configuration file**

   Define the preprocessing steps and parameters in a YAML configuration file.
   You can use the benchmark configuration file shown above as a starting point.

3. **Run the benchmark**

   Use the ``-bt`` option to provide the benchmark configuration as a JSON
   string, and use ``-c`` to provide the Paidiverpy pipeline configuration.

   For example, to benchmark a local Dask cluster:

   .. code-block:: bash

      paidiverpy -bt '{"cluster_type":"local","n_workers":[1,8,16],"threads_per_worker":[1,8,16],"memory_limit":[32],"n_jobs":[2]}' \
        -c <path_to_your_config_file>

   This command runs the benchmark using a local Dask cluster. It varies the
   number of workers, threads per worker, and memory limit while keeping
   ``n_jobs`` fixed at ``2``.

   To benchmark thread-based execution without creating a Dask cluster, vary only
   ``n_jobs``:

   .. code-block:: bash

      paidiverpy -bt '{"n_jobs":[1,2,4,8,16]}' \
        -c <path_to_your_config_file>

4. **Review the outputs**

   The benchmark writes the results to a JSON file named according to the cluster
   type and timestamp, for example:

   .. code-block:: text

      benchmark_results_local_20250916_103045.json

   The benchmark utility also generates a PNG plot and an HTML report for visual
   inspection.

   When running through Slurm, these output files are written after the batch job
   finishes, in the directory from which the CLI was launched.

   The HTML report requires Plotly. If Plotly is not already installed in your
   environment, install it with:

   .. code-block:: bash

      pip install plotly
