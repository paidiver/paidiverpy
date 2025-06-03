.. _guide_performance:

Performance
===========

Overview
--------

In **paidiverpy**, users can execute the pipeline either sequentially or in parallel. Parallel execution is powered by **Dask**, a flexible parallel computing library. This section provides an overview of how to configure and measure performance when running pipelines in parallel.

Configuring Pipeline Execution
------------------------------

Pipeline execution mode (sequential or parallel) is controlled via the configuration file. Two parameters determine the execution method:

- ``n_jobs``: Controls the number of jobs for local execution.
- ``client``: Configures execution in a High-Performance Computing (HPC) environment.

Local Execution
---------------

The ``n_jobs`` parameter specifies the number of parallel jobs. By default, ``n_jobs`` is set to ``1``, meaning the pipeline runs sequentially. To enable parallel execution:

- Set ``n_jobs`` to a number greater than ``1``, up to the number of available CPU cores.
- Use ``n_jobs: -1`` to automatically match the number of jobs to the total CPU cores.

Example Configuration File for Local Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'
      n_jobs: -1

    steps:
      # Define pipeline steps

In the example above, the pipeline runs in parallel using all available CPU cores. To disable parallel execution, set ``n_jobs`` to ``1`` or omit it.

HPC Execution
-------------

To run the pipeline on an HPC cluster, use the ``client`` parameter. By default, ``client`` is ``None``, which means sequential execution.

You can configure ``client`` for two common use cases:

1. LocalCluster (Dask)
^^^^^^^^^^^^^^^^^^^^^^

To create a **LocalCluster**, configure the ``client`` parameter as follows:

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'
      n_jobs: 2
      client:
        cluster_type: "local"
        params:
          n_workers: 1
          threads_per_worker: 4
          memory_limit: "4GB"
      track_changes: False

    steps:
      # Define pipeline steps

The ``params`` block corresponds to parameters for the Dask **LocalCluster** class, where you can specify the number of workers, threads per worker, and memory limits.

You also need to specify the number of jobs in the ``general`` section. In this example, ``n_jobs: 2`` means that the Client will be scaled to 2 workers.

2. SLURMCluster (Dask-Jobqueue)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Note

  IMPORTANT: This feature will be available in the upcoming release.

To create a **SLURMCluster**, configure the ``client`` parameter as follows:

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'
      client:
        cluster_type: "slurm"
        params:
          n_workers: 1
          threads_per_worker: 4
          memory_limit: "4GB"
          job_extra: ["--partition=standard", "--time=00:30:00"]
      track_changes: False

    steps:
      # Define pipeline steps

Here, the ``params`` block maps to parameters for the Dask **SLURMCluster** class. You can specify workers, threads, memory limits, and additional job options.

You also need to specify the number of jobs in the ``general`` section. In this example, ``n_jobs: 2`` means that the Client will be scaled to 2 workers.

Key Considerations
------------------

1. **Sequential Dependency**: Pipeline parallelism operates within individual steps, not across steps. Each step must complete before the next begins, as the output of one step serves as the input for the next.

2. **Temporary Directories**: For HPC execution, it is important to set ``track_changes: False``, which means the pipeline does not track intermediate changes. This setting is essential for HPC environments in order to speed up execution and avoid unnecessary file transfers.

Examples and Resources
----------------------

- **Configuration Files**: Find example configuration files for parallel execution in the `GitHub repository <https://github.com/paidiver/paidiverpy/tree/main/examples/config_files>`_.

- **Interactive Examples**: Explore example notebooks with custom algorithms in the :ref:`gallery` section.
