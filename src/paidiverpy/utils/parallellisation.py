"""Module for parallelisation utilities."""

import logging
import multiprocessing
import os
import dask
import dask.config
from dask.distributed import Client
from dask.distributed import LocalCluster
from paidiverpy.models.general_config import GeneralConfig

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


def parse_parallellisation_params(config: GeneralConfig | None) -> Client | None:
    """Parse the client configuration.

    Args:
        config (GeneralConfig | None): Client configuration.

    Returns:
        dask.distributed.Client | None: Dask client or None if no client is configured.
    """
    config_client = config.local_cluster if config else None
    dask_config_kwargs = config.dask_config_kwargs if config else None
    update_dask_config(dask_config_kwargs)
    if config_client is None:
        return None
    cluster = LocalCluster(**config_client)
    n_jobs = config.n_jobs if config else 1

    cluster.scale(n_jobs)
    client = Client(cluster)
    logger.info("Created LocalCluster with Client: %s", client.dashboard_link)
    return client
