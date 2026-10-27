paidiverpy.utils.parallellisation
=================================

.. py:module:: paidiverpy.utils.parallellisation

.. autoapi-nested-parse::

   Module for parallelisation utilities.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.parallellisation.get_n_jobs
   paidiverpy.utils.parallellisation.update_dask_config
   paidiverpy.utils.parallellisation.parse_parallellisation_params


Module Contents
---------------

.. py:function:: get_n_jobs(n_jobs: int) -> int

   
   Determine the number of jobs based on n_jobs parameter.

   Uses SLURM_CPUS_ON_NODE when inside a Slurm allocation so that only the
   CPUs actually allocated to the job are used, not all CPUs visible on the node.

   :param n_jobs: The number of n_jobs.
   :type n_jobs: int

   :returns: The number of jobs to use.
   :rtype: int















   ..
       !! processed by numpydoc !!

.. py:function:: update_dask_config(dask_config_kwargs: dict) -> None

   
   Update the Dask configuration.

   :param dask_config_kwargs: Dask configuration keyword arguments.
   :type dask_config_kwargs: dict















   ..
       !! processed by numpydoc !!

.. py:function:: parse_parallellisation_params(config: paidiverpy.models.general_config.GeneralConfig | None) -> dask.distributed.Client | None

   
   Parse the client configuration.

   :param config: Client configuration.
   :type config: GeneralConfig | None

   :returns: Dask client or None if no client is configured.
   :rtype: dask.distributed.Client | None















   ..
       !! processed by numpydoc !!

