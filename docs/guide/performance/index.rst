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

For Slurm execution, Paidiverpy should be launched from within a Slurm batch
allocation. In this mode, Slurm is responsible for reserving the compute
resources, and Paidiverpy uses those resources internally through threads or a
local Dask scheduler.

Paidiverpy does not submit nested Slurm jobs from inside the pipeline. Instead,
you submit an ``sbatch`` wrapper script that:

- requests the required CPUs, memory, walltime, partition, and account;
- activates the Python environment containing Paidiverpy;
- runs the ``paidiverpy`` command using the chosen configuration file.

Use the template available at
`examples/slurm/paidiverpy.sbatch <https://github.com/paidiver/paidiverpy/tree/main/examples/slurm/paidiverpy.sbatch>`_
as a reference for the job wrapper.

A typical Slurm-oriented configuration looks like this:

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

The ``n_jobs`` value should be consistent with the CPUs requested in the Slurm
batch file. The Slurm allocation is treated as the source of truth for available
resources.

For example, if the batch script requests 16 CPUs, you can either set:

.. code-block:: yaml

    n_jobs: 16

or use:

.. code-block:: yaml

    n_jobs: -1

to allow Paidiverpy to use all CPUs made available inside the allocation.

Avoid requesting more workers in the Paidiverpy configuration than the number of
CPUs allocated by Slurm. Doing so can oversubscribe the node, increase memory
pressure, and reduce overall performance.

For batch or HPC runs, it is also recommended to set:

.. code-block:: yaml

    track_changes: False

This disables tracking of intermediate pipeline changes, which can reduce I/O
overhead and avoid unnecessary file transfers on shared filesystems.

A minimal Slurm wrapper typically follows this pattern:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=paidiverpy
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=64G
    #SBATCH --time=04:00:00
    #SBATCH --partition=<partition>
    #SBATCH --account=<account>

    source /path/to/venv/bin/activate

    paidiverpy -c /path/to/config.yaml

In summary, the Slurm batch script controls the resources, while ``n_jobs``
controls how much parallel work Paidiverpy attempts to perform within those
resources.

Key Considerations
------------------

1. **Sequential Dependency**: Pipeline parallelism operates within individual steps, not across steps. Each step must complete before the next begins, as the output of one step serves as the input for the next.

2. **Temporary Directories**: For batch or HPC execution, it is important to set ``track_changes: False``, which means the pipeline does not track intermediate changes. This setting is essential in order to speed up execution and avoid unnecessary file transfers.

Examples and Resources
----------------------

- **Configuration Files**: Find example configuration files for parallel execution in the `GitHub repository <https://github.com/paidiver/paidiverpy/tree/main/examples/config_files>`_.

- **Interactive Examples**: Explore example notebooks with custom algorithms in the :ref:`gallery` section.
