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
- ``local_cluster``: Configures a local Dask cluster when you want Dask-backed parallel execution.

Slurm batch execution is handled outside the pipeline config by submitting an ``sbatch`` wrapper. The batch job then launches Paidiverpy inside the Slurm allocation and the pipeline runs without creating nested Slurm jobs.

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

Local Dask Cluster
------------------

To create a **LocalCluster**, configure the ``local_cluster`` section as follows:

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'
      n_jobs: 2
      local_cluster:
        n_workers: 1
        threads_per_worker: 4
        memory_limit: "4GB"
      track_changes: False

    steps:
      # Define pipeline steps

The ``local_cluster`` section corresponds to the Dask **LocalCluster** class. You can specify the number of workers, threads per worker, and memory limits.

The ``n_jobs`` parameter still controls how much work the pipeline tries to perform in parallel. In local runs, set ``n_jobs`` to ``-1`` if you want to use all available CPUs.

Slurm Batch Submission
----------------------

For Slurm execution, you submit a batch job that runs Paidiverpy inside the allocated resources. The batch job should activate the environment and execute the pipeline with the thread scheduler, which avoids creating nested Slurm jobs.

Use the template available at `examples/slurm/paidiverpy.sbatch <https://github.com/paidiver/paidiverpy/tree/main/examples/slurm/paidiverpy.sbatch>`_ as a reference for the job wrapper. The important part is that the batch job activates the environment, runs ``paidiverpy`` inside the allocation, and leaves the inner pipeline to use threads or a local Dask scheduler.

Example batch configuration:

.. code-block:: yaml

    general:
      input_path: '/input/data/path/'
      output_path: '/output/data/path/'
      metadata_path: '/metadata/path/metadata.json'
      metadata_type: 'IFDO'
      image_open_args: 'JPG'
      n_jobs: -1
      track_changes: False

    steps:
      # Define pipeline steps

In the Slurm batch file, set the requested CPUs, memory, walltime, queue, and account to match the workload you want to benchmark. The benchmark helper then writes the JSON and plot files once the batch job finishes.

If you allocate CPUs through Slurm, keep ``n_jobs`` consistent with the CPUs you requested. The pipeline now treats the batch allocation as the source of truth.
For example, if you request 16 CPUs in the batch job, set ``n_jobs: 16`` or ``n_jobs: -1`` in the configuration file to fully utilize the allocated resources.

Key Considerations
------------------

1. **Sequential Dependency**: Pipeline parallelism operates within individual steps, not across steps. Each step must complete before the next begins, as the output of one step serves as the input for the next.

2. **Temporary Directories**: For batch or HPC execution, it is important to set ``track_changes: False``, which means the pipeline does not track intermediate changes. This setting is essential in order to speed up execution and avoid unnecessary file transfers.

Examples and Resources
----------------------

- **Configuration Files**: Find example configuration files for parallel execution in the `GitHub repository <https://github.com/paidiver/paidiverpy/tree/main/examples/config_files>`_.

- **Interactive Examples**: Explore example notebooks with custom algorithms in the :ref:`gallery` section.
