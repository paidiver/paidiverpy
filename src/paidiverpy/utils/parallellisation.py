"""Module for parallelisation utilities."""

import logging
import multiprocessing
import os
from typing import Any
import dask
import dask.config
from dask.distributed import Client
from dask.distributed import LocalCluster
from paidiverpy.models.client_params import ClientParams

logger = logging.getLogger("paidiverpy")


def get_n_jobs(n_jobs: int) -> int:
    """Determine the number of jobs based on n_jobs parameter.

    Uses SLURM_CPUS_ON_NODE when inside a Slurm allocation so that only the
    CPUs actually allocated to the job are used, not all CPUs visible on the node.

    Args:
        n_jobs (int): The number of n_jobs.

    Returns:
        int: The number of jobs to use.
    """
    available = int(os.environ.get("SLURM_CPUS_ON_NODE") or multiprocessing.cpu_count())
    if n_jobs == -1:
        return available
    if n_jobs > 1:
        return min(n_jobs, available)
    return 1


def update_dask_config(dask_config_kwargs: dict) -> None:
    """Update the Dask configuration.

    Args:
        dask_config_kwargs (dict): Dask configuration keyword arguments.
    """
    if dask_config_kwargs is not None:
        dask.config.set(dask_config_kwargs)
        logger.info("Updated dask configuration settings")


def parse_dask_job(job: dict, n_jobs: int) -> tuple[Client, list[str]] | Client:
    """Parse the Dask job configuration.

    Args:
        job (dict): Job configuration.
        n_jobs (int): Number of jobs.

    Returns:
        tuple[Client, list[str]] | Client: Dask client and job IDs for Slurm, or just client for local.
    """
    update_dask_config(job.get("dask_config_kwargs"))
    params = dict(job.get("params") or {})
    # requested_cluster_type = job.get("cluster_type")

    cluster = LocalCluster(**params)
    cluster_type = "LocalCluster"

    cluster.scale(n_jobs)
    client = Client(cluster)
    logger.info("Created %s with Client: %s", cluster_type, client.dashboard_link)
    return client


def parse_client(config_client: dict[str, Any] | ClientParams | None, n_jobs: int) -> Client | None:
    """Parse the client configuration.

    Args:
        config_client (dict | ClientParams | None): Client configuration.
        n_jobs (int): Number of jobs.

    Returns:
        dask.distributed.Client | None: Dask client or None if no client is configured.
    """
    if config_client is None:
        return None
    config_client = config_client.to_dict() if isinstance(config_client, ClientParams) else config_client
    cluster_type = config_client.get("cluster_type")
    if cluster_type == "slurm":
        result = parse_dask_job(config_client, n_jobs)
        # result is a tuple (client, job_ids) for slurm
        client = result[0] if isinstance(result, tuple) else result
    elif cluster_type == "local":
        client = parse_dask_job(config_client, n_jobs)
    else:
        client = None
    return client
