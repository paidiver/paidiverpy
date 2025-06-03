"""Module for parallelisation utilities."""

import logging
import multiprocessing
import dask
import dask.config
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster
from paidiverpy.models.client_params import ClientParams


def get_n_jobs(n_jobs: int) -> int:
    """Determine the number of jobs based on n_jobs parameter.

    Args:
        n_jobs (int): The number of n_jobs.

    Returns:
        int: The number of jobs to use.
    """
    if n_jobs == -1:
        return multiprocessing.cpu_count()
    if n_jobs > 1:
        return min(n_jobs, multiprocessing.cpu_count())
    return 1


def update_dask_config(dask_config_kwargs: dict) -> None:
    """Update the Dask configuration.

    Args:
        dask_config_kwargs (dict): Dask configuration keyword arguments.
    """
    if dask_config_kwargs is not None:
        dask.config.set(dask_config_kwargs)
        logging.info("Updated dask configuration settings")


def parse_dask_job(job: dict, n_jobs: int) -> Client:
    """Parse the Dask job configuration.

    Args:
        job (dict): Job configuration.
        n_jobs (int): Number of jobs.

    Returns:
        dask.distributed.Client: Dask client.
    """
    update_dask_config(job.get("dask_config_kwargs"))
    if job.get("cluster_type") == "slurm":
        cluster = SLURMCluster(**job.get("params"))
        cluster_type = "SLURMCluster"
        job_id = None
    elif job.get("cluster_type") == "local":
        cluster = LocalCluster(**job.get("params"))
        cluster_type = "LocalCluster"
        job_id = None
    cluster.scale(n_jobs)
    client = Client(cluster)
    logging.info("Created %s with Client: %s", cluster_type, client.dashboard_link)
    if cluster_type == "SLURMCluster":
        return (client, job_id)
    return client


def get_client(config_client: dict | ClientParams | None, n_jobs: int) -> Client:
    """Parse the client configuration.

    Args:
        config_client (dict | ClientParams | None): Client configuration.
        n_jobs (int): Number of jobs.

    Returns:
        dask.distributed.Client: Dask client.
    """
    if config_client is None:
        return None
    config_client = config_client.to_dict() if isinstance(config_client, ClientParams) else config_client
    job_id = None
    cluster_type = config_client.get("cluster_type")
    if cluster_type == "slurm":
        client, job_id = parse_dask_job(config_client, n_jobs)
    elif cluster_type == "local":
        client = parse_dask_job(config_client, n_jobs)
    if cluster_type == "slurm":
        return client, job_id
    return client
