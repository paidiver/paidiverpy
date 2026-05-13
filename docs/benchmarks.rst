.. _benchmarks:

Benchmarks
==========

This section provides an overview of the benchmarks conducted to evaluate the performance of the **Paidiverpy** library.

The benchmarks focus on execution speed, memory usage, and scalability across different configurations. By publishing these results, our goal is to give users insights into the performance characteristics of Paidiverpy and to provide guidance on how to tune resources for their own use cases.
Paidiverpy also makes it easy to run **custom benchmark tests** using the
`benchmark_test.py <https://github.com/paidiver/paidiverpy/blob/dev/src/paidiverpy/utils/benchmark/benchmark_test.py>`_
script. Instructions for running your own benchmarks are provided at the end of this section.


About the tests
---------------

For our benchmark experiments, we used the **benthic_ifdo** dataset, described in the :ref:`example data <example_data>` section. This dataset contains images and metadata from the Haig Fras area in the UK, collected in 2012.
It is publicly available through the British Oceanographic Data Centre (BODC) and can be accessed `here <https://www.bodc.ac.uk/data/published_data_library/catalogue/10.5285/093edbc7-3552-3d35-e063-6c86abc099d5/>`_.
For benchmarking, we used a **subset** of this dataset (≈290 MB), which contains 100 images with varying resolutions.

The pipeline configuration for preprocessing is defined in the following YAML file:

.. literalinclude:: ../src/paidiverpy/utils/benchmark/config_benchmark.yml

This configuration applies several preprocessing steps, mainly from the **ColourLayer** and **ConvertLayer** components. The ``benchmark_test.py`` script executes this pipeline, measuring performance across multiple configurations by varying the number of threads, workers, and memory resources depending on the cluster type.

Cluster workflow
----------------

Paidiverpy now uses a simple two-level model for clusters:

- For **interactive or local runs**, configure ``general.local_cluster`` and let the pipeline create a local Dask cluster when needed.
- For **Slurm benchmarking**, submit one outer batch job and let that job run the benchmark internally. The batch job uses the CPUs requested in the sbatch file and executes the benchmark with the thread scheduler, which avoids creating extra nested Slurm jobs.

The Slurm batch wrapper used by the benchmark workflow is shown in ``examples/slurm/benchmark.sbatch``.


Test configurations
-------------------

The benchmarks were run under different execution modes:

- **Serial execution**: with ``n_jobs`` set to 1.

  .. code-block:: yaml

     n_jobs: [1]

- **Thread-based execution**: varying only the ``n_jobs`` parameter:

  .. code-block:: yaml

     n_jobs: [2, 4, 8, 16]

- **Local Dask cluster**: varying the number of workers, threads per worker, and memory limit:

  .. code-block:: yaml

     n_jobs: [1, 2, 4, 8, 16]
       local_cluster:
       n_workers: [1, 8, 16]
       threads_per_worker: [1, 8, 16]
       memory_limit: [32]

- **Slurm batch submission**: varying the sbatch resources and benchmark scale while the inner benchmark run uses the thread scheduler:

   .. code-block:: yaml

       cluster_type: slurm
       cores: [16]
       memory: [64]
       walltime: "12:00:00"
       queue: "standard"
       n_jobs: [1, 2, 4, 8, 16]

Results
-------

This section is still under development. Preliminary results indicate that Paidiverpy scales well with increased resources, particularly in multi-threaded and distributed configurations.

We are planning to include more detailed analyses and visualisations in future updates.

.. The detailed benchmark results are stored in a JSON file, available here:
.. `benchmark.json <https://github.com/paidiver/paidiverpy/blob/dev/src/paidiverpy/utils/benchmark/benchmark.json>`_

.. This file includes execution time and memory usage for each preprocessing run.

.. A graphical representation of the results is shown below:

.. .. image:: _static/benchmark_plot.png
..    :width: 600px
..    :alt: Benchmark Plot


How to run your own benchmarks
------------------------------

You can run benchmarks tailored to your dataset and pipeline configuration by following these steps:

1. **Prepare a dataset**
   Ensure your dataset is supported by Paidiverpy and ready for preprocessing.

2. **Create a configuration file**
   Define your preprocessing steps and parameters in a YAML configuration file. You may use the provided benchmark configuration files as a starting point.

3. **Run the benchmark script**
   Launch the benchmark with the CLI, specifying the cluster configuration. For example:

   .. code-block:: bash

      paidiverpy -bt '{"cluster_type":"local","n_workers":[1,8,16],"threads_per_worker":[1,8,16],"memory_limit":[32],"n_jobs":[2]}' \
      -c <path_to_your_config_file>

   For Slurm submission, use the same CLI with ``cluster_type`` set to ``slurm`` and resource values such as ``cores``, ``memory``, ``queue``, and ``walltime``. The CLI will generate the batch script from the example template and submit the job for you:

   .. code-block:: bash

      paidiverpy -bt '{"cluster_type":"slurm","cores":[16],"memory":[64],"walltime":"12:00:00","queue":"standard","n_jobs":[1,2,4,8,16]}' \
      -c <path_to_your_config_file>

   The batch job will execute the pipeline multiple times, varying the parameters, and record performance results.

4. **Analyze results**
   Results are stored in a JSON file named according to the cluster type and timestamp, e.g.:

   .. code-block:: text

      benchmark_results_local_20250916_103045.json

   Additionally, the script generates a PNG plot and an HTML output for visual inspection. If you submit the benchmark through Slurm, the batch job writes those files after the run finishes in the directory where you launched the CLI. You may need to install plotly to generate the HTML output:

   .. code-block:: bash

      pip install plotly
