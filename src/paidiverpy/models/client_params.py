"""Client parameters module."""

from typing import Literal
from pydantic import Field
from paidiverpy.utils.base_model import BaseModel


class ClientParams(BaseModel):
    """Client parameters class.

    This class is used to define the client parameters for Dask parallel processing.

    """

    cluster_type: Literal["local", "slurm"] = Field(default="local", description="Type of cluster")
    params: dict = Field({"n_workers": 0}, description="Parameters for the cluster")
