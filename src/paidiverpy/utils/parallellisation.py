"""Module for parallelisation utilities."""

import logging
import multiprocessing
import dask
import dask.config
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


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


def parse_dask_job(job: dict) -> Client:
    """Parse the Dask job configuration.

    Args:
        job (dict): Job configuration.

    Returns:
        dask.distributed.Client: Dask client.
    """
    update_dask_config(job.get("dask_config_kwargs"))
    if job.get("type") == "slurm":
        cluster = SLURMCluster(job.get("job_cluster_kwargs"))
        cluster_type = "SLURMCluster"
    else:
        cluster = LocalCluster(job.get("job_cluster_kwargs"))
        cluster_type = "LocalCluster"
    client = Client(cluster)
    logging.info("Created %s with Client: %s", cluster_type, client.dashboard_link)
    return client


def get_client(config_client: dict) -> Client:
    """Parse the client configuration.

    Args:
        config_client (dict): Client configuration.

    Returns:
        dask.distributed.Client: Dask client.
    """
    if config_client is None:
        return None
    cluster_type = config_client.get("cluster_type")
    if cluster_type == "slurm":
        client = parse_dask_job(config_client)
    if cluster_type == "dask":
        client = parse_dask_job(config_client)
    else:
        msg = f"Job type {cluster_type} not supported."
        raise ValueError(msg)
    return client
