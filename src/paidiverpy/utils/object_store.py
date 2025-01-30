""" This module contains utility functions for interacting with object storage. """
import os
from dotenv import load_dotenv
load_dotenv()

def define_storage_options(path) -> dict:
    """ Define storage options for reading metadata file.

    Returns:
        dict: Storage options for reading metadata file.
    """
    storage_options = {}
    if str(path).startswith("s3://"):
        os_token = os.getenv("OS_TOKEN")
        os_secret = os.getenv("OS_SECRET")
        os_endpoint = os.getenv("OS_ENDPOINT")
        if not os_token or not os_secret:
            msg = "You are trying to access an S3 bucket without providing the necessary credentials."
            raise ValueError(msg)

        storage_options = {
            "key": os_token,
            "secret": os_secret,
        }
        if os_endpoint:
            storage_options["client_kwargs"] = {}
            storage_options["client_kwargs"]["endpoint_url"] = os_endpoint
    return storage_options
