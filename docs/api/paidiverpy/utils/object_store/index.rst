paidiverpy.utils.object_store
=============================

.. py:module:: paidiverpy.utils.object_store

.. autoapi-nested-parse::

   This module contains utility functions for interacting with object storage.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.object_store.define_storage_options
   paidiverpy.utils.object_store.get_file_from_bucket
   paidiverpy.utils.object_store.create_client
   paidiverpy.utils.object_store.check_create_bucket_exists
   paidiverpy.utils.object_store.upload_file_to_bucket
   paidiverpy.utils.object_store.path_is_remote


Module Contents
---------------

.. py:function:: define_storage_options(path: str | pathlib.Path) -> dict

   
   Define storage options for reading metadata file.

   :param path: Path to the metadata file.
   :type path: str | Path

   :returns: Storage options for reading metadata file.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: get_file_from_bucket(file_path: str, storage_options: dict | None = None) -> bytes

   
   Get a file from an object store bucket.

   :param file_path: Full S3 path (e.g., "s3://my-bucket/path/to/image.png").
   :type file_path: str
   :param storage_options: Storage options for reading metadata file. Defaults to None.
   :type storage_options: dict

   :returns: The file content.
   :rtype: bytes















   ..
       !! processed by numpydoc !!

.. py:function:: create_client() -> boto3.client

   
   Create a boto3 client for S3.

   :returns: A boto3 client for S3.
   :rtype: boto3.client















   ..
       !! processed by numpydoc !!

.. py:function:: check_create_bucket_exists(bucket_name: str, client: boto3.client, logger: logging.Logger) -> None

   
   Check if a bucket exists.

   :param bucket_name: The name of the bucket.
   :type bucket_name: str
   :param client: The boto3 client for S3.
   :type client: boto3.client
   :param logger: The logger to log messages.
   :type logger: logging.Logger















   ..
       !! processed by numpydoc !!

.. py:function:: upload_file_to_bucket(file_obj: io.BytesIO, output_path: str, client: boto3.client) -> None

   
   Upload an in-memory file to an object store bucket.

   :param file_obj: In-memory file object.
   :type file_obj: io.BytesIO
   :param output_path: Full S3 path (e.g., "s3://my-bucket/path/to/image.png").
   :type output_path: str
   :param client: The boto3 client for S3.
   :type client: boto3.client















   ..
       !! processed by numpydoc !!

.. py:function:: path_is_remote(path: str) -> bool

   
   Check if the path is a remote path.

   :param path: The path to check.
   :type path: str

   :returns: True if the path is remote, False otherwise.
   :rtype: bool















   ..
       !! processed by numpydoc !!

