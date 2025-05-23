"""This module contains utility functions for interacting with object storage."""

import io
import logging
import os
from pathlib import Path
import boto3
import botocore
import requests
from dotenv import load_dotenv

load_dotenv()


def define_storage_options(path: str | Path) -> dict:
    """Define storage options for reading metadata file.

    Args:
        path (str | Path): Path to the metadata file.

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
            "aws_access_key_id": os_token,
            "aws_secret_access_key": os_secret,
            "service_name": "s3",
        }
        if os_endpoint:
            storage_options["endpoint_url"] = os_endpoint
    return storage_options


def get_file_from_bucket(file_path: str, storage_options: dict | None = None) -> bytes:
    """Get a file from an object store bucket.

    Args:
        file_path (str): Full S3 path (e.g., "s3://my-bucket/path/to/image.png").
        storage_options (dict): Storage options for reading metadata file. Defaults to None.

    Returns:
        bytes: The file content.
    """
    if file_path.startswith("s3://"):
        s3_path = file_path[5:]
        bucket_name, key = s3_path.split("/", 1)
        s3_client = boto3.client(**storage_options)
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return response["Body"].read()
    response = requests.get(file_path, stream=True, timeout=10)
    response.raise_for_status()
    return response.content


def create_client() -> boto3.client:
    """Create a boto3 client for S3.

    Returns:
        boto3.client: A boto3 client for S3.
    """
    storage_options = define_storage_options("s3://")
    return boto3.client(**storage_options)


def check_create_bucket_exists(bucket_name: str, client: boto3.client, logger: logging.Logger) -> None:
    """Check if a bucket exists.

    Args:
        bucket_name (str): The name of the bucket.
        client (boto3.client): The boto3 client for S3.
        logger (logging.Logger): The logger to log messages.
    """
    exists = True
    try:
        client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError:
        exists = False
    if not exists:
        logger.info("Creating bucket %s.", bucket_name)
        client.create_bucket(Bucket=bucket_name)


def upload_file_to_bucket(file_obj: io.BytesIO, output_path: str, client: boto3.client) -> None:
    """Upload an in-memory file to an object store bucket.

    Args:
        file_obj (io.BytesIO): In-memory file object.
        output_path (str): Full S3 path (e.g., "s3://my-bucket/path/to/image.png").
        client (boto3.client): The boto3 client for S3.
    """
    s3_path = output_path[5:]
    bucket_name, key = s3_path.split("/", 1)
    client.put_object(Body=file_obj.getvalue(), Bucket=bucket_name, Key=key)


def path_is_remote(path: str) -> bool:
    """Check if the path is a remote path.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is remote, False otherwise.
    """
    return str(path).startswith(("http://", "https://", "s3://"))
