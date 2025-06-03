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
   paidiverpy.utils.parallellisation.parse_dask_job
   paidiverpy.utils.parallellisation.get_client


Module Contents
---------------

.. py:function:: get_n_jobs(n_jobs: int) -> int

   
   Determine the number of jobs based on n_jobs parameter.

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

.. py:function:: parse_dask_job(job: dict, n_jobs: int) -> dask.distributed.Client

   
   Parse the Dask job configuration.

   :param job: Job configuration.
   :type job: dict
   :param n_jobs: Number of jobs.
   :type n_jobs: int

   :returns: Dask client.
   :rtype: dask.distributed.Client















   ..
       !! processed by numpydoc !!

.. py:function:: get_client(config_client: dict | paidiverpy.models.client_params.ClientParams | None, n_jobs: int) -> dask.distributed.Client

   
   Parse the client configuration.

   :param config_client: Client configuration.
   :type config_client: dict | ClientParams | None
   :param n_jobs: Number of jobs.
   :type n_jobs: int

   :returns: Dask client.
   :rtype: dask.distributed.Client















   ..
       !! processed by numpydoc !!

