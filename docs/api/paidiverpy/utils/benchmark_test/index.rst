paidiverpy.utils.benchmark_test
===============================

.. py:module:: paidiverpy.utils.benchmark_test

.. autoapi-nested-parse::

   This module contains functions to run the benchmark test.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.benchmark_test.benchmark_task
   paidiverpy.utils.benchmark_test.plot_results
   paidiverpy.utils.benchmark_test.update_yaml
   paidiverpy.utils.benchmark_test.benchmark_threads
   paidiverpy.utils.benchmark_test.benchmark_local
   paidiverpy.utils.benchmark_test.benchmark_slurm
   paidiverpy.utils.benchmark_test.benchmark_handler


Module Contents
---------------

.. py:function:: benchmark_task(configuration_file: str, logger: logging.Logger) -> None

   
   Run the benchmark task.

   :param configuration_file: The path to the configuration file.
   :type configuration_file: str
   :param logger: The logger to log messages.
   :type logger: logging.Logger















   ..
       !! processed by numpydoc !!

.. py:function:: plot_results(results: list, cluster_type: str, filename: str) -> None

   
   Plot the benchmark results.

   :param results: The list of benchmark results.
   :type results: list
   :param cluster_type: The cluster type.
   :type cluster_type: str
   :param filename: The filename to save the plot.
   :type filename: str















   ..
       !! processed by numpydoc !!

.. py:function:: update_yaml(file_path: str, cluster_type: str, output_file: str, n_jobs: int, **kwargs: dict) -> str

   
   Update the YAML file with new benchmarking parameters and save it.

   :param file_path: The path to the configuration file.
   :type file_path: str
   :param cluster_type: The cluster type.
   :type cluster_type: str
   :param output_file: The output file path.
   :type output_file: str
   :param n_jobs: The number of jobs.
   :type n_jobs: int
   :param \*\*kwargs: The benchmarking parameters. It should be a dictionary with the following:
                      For LocalCluster:
                      - workers (int): The number of workers.
                      - threads (int): The number of threads.
                      - memory (int): The memory limit.
                      For SLURM:
                      - cores (int): The number of cores.
                      - processes (int): The number of processes.
                      - memory (int): The memory limit.
                      - walltime (str): The walltime.
                      - queue (str): The queue name.

   :returns: The output file path.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: benchmark_threads(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> list

   
   Handle the benchmark test for LocalCluster.

   :param benchmark_params: The benchmark parameters.
   :type benchmark_params: dict
   :param configuration_file: The path to the configuration files.
   :type configuration_file: str
   :param logger: The logger to log messages.
   :type logger: logging.Logger

   :returns: The benchmark results.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: benchmark_local(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> list

   
   Handle the benchmark test for LocalCluster.

   :param benchmark_params: The benchmark parameters.
   :type benchmark_params: dict
   :param configuration_file: The path to the configuration files.
   :type configuration_file: str
   :param logger: The logger to log messages.
   :type logger: logging.Logger

   :returns: The benchmark results.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: benchmark_slurm(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> list

   
   Handle the benchmark test for SLURM.

   :param benchmark_params: The benchmark parameters.
   :type benchmark_params: dict
   :param configuration_file: The path to the configuration files.
   :type configuration_file: str
   :param logger: The logger to log messages.
   :type logger: logging.Logger

   :returns: The benchmark results.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: benchmark_handler(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> None

   
   Handle the benchmark test.

   :param benchmark_params: The benchmark parameters.
   :type benchmark_params: dict
   :param configuration_file: The path to the configuration files.
   :type configuration_file: str
   :param logger: The logger to log messages.
   :type logger: logging.Logger















   ..
       !! processed by numpydoc !!

