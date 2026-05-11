"""Utility classes and functions for testing purposes."""

import io
from types import SimpleNamespace
import botocore


class DummyLogger:
    """Simple dummy logger that records messages for testing purposes."""

    def __init__(self):
        self.messages = []

    def info(self, *args: object, **kwargs: object) -> None:
        """Record an info message."""
        self.messages.append(("info", args, kwargs))

    def warning(self, *args: object, **kwargs: object) -> None:
        """Record a warning message."""
        self.messages.append(("warning", args, kwargs))

    def error(self, *args: object, **kwargs: object) -> None:
        """Record an error message."""
        self.messages.append(("error", args, kwargs))

    def debug(self, *args: object, **kwargs: object) -> None:
        """Record a debug message."""
        self.messages.append(("debug", args, kwargs))


class DummyStep:
    """Simple dummy step for testing purposes."""

    def __init__(self, name: str, step_name: str):
        self.name = name
        self.step_name = step_name

    def to_dict(self):
        """Convert the dummy step to a dictionary representation."""
        return {"name": self.name, "step_name": self.step_name, "mode": "fixed", "params": {"value": 1}}


class FakeCluster:
    """Simple fake cluster for testing purposes."""

    def __init__(self, **kwargs: object):
        self.kwargs = kwargs
        self.scaled = None

    def scale(self, value: object):
        """Simulate scaling the cluster by setting the scaled attribute."""
        self.scaled = value


class FakePath:
    """Simple fake path for testing purposes."""

    def __init__(self, value: str):
        self._value = value
        self.name = value.split("/")[-1]

    def open(self) -> io.StringIO:
        """Simulate opening a file by returning a StringIO with invalid JSON content."""
        return io.StringIO("{ invalid json")

class FakeClient:
    """Simple fake client for testing purposes."""

    def __init__(self, cluster: FakeCluster):
        self.cluster = cluster
        self.dashboard_link = "http://fake-dashboard"


class DummyTqdm:
    """Small context manager replacement for tqdm in tests."""

    def __init__(self, *args: object, **kwargs: object):  # noqa: ARG002
        self.total = kwargs.get("total")
        self.updated = 0

    def __enter__(self):
        """Return self for context manager usage."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> bool:
        """Exit the context manager.

        Args:
            exc_type (type[BaseException] | None): The type of the exception, if any.
            exc (BaseException | None): The exception instance, if any.
            tb (object): The traceback, if any.

        Returns:
            bool: False to indicate that exceptions should not be suppressed.
        """
        return False

    def update(self, value: int):
        """Simulate tqdm update by incrementing the internal counter.

        Args:
            value (int): The amount to increment the counter by.
        """
        self.updated += value


class DummyResponse:
    """Dummy requests response for download/get tests."""

    def __init__(self, chunks: list[bytes], content: bytes = b"", status_ok: bool = True):
        self._chunks = chunks
        self.content = content
        self.headers = {"content-length": str(sum(len(c) for c in chunks))}
        self._status_ok = status_ok

    def raise_for_status(self):
        """Simulate raise_for_status method of requests response."""
        if not self._status_ok:
            msg = "request failed"
            raise RuntimeError(msg)

    def iter_content(self, _block_size: int):
        """Simulate iter_content method of requests response.

        Args:
            _block_size (int): The block size for iteration (ignored in dummy).

        Yields:
            bytes: Chunks of data.
        """
        yield from self._chunks


class DummyS3Client:
    """Simple fake S3 client used by object-store tests."""

    def __init__(self):
        self.created_bucket = None
        self.put_calls = []
        self.should_raise_head = False

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        """Simulate get_object method of S3 client.

        Args:
            Bucket (str): The name of the bucket.
            Key (str): The key of the object.

        Returns:
            dict: A dictionary with a 'Body' key containing a SimpleNamespace with a read method.
        """
        assert Bucket == "bucket"
        assert Key == "path/to/file.bin"
        return {"Body": SimpleNamespace(read=lambda: b"s3-bytes")}

    def head_bucket(self, Bucket: str) -> dict:  # noqa: N803
        """Simulate head_bucket method of S3 client.

        Args:
            Bucket (str): The name of the bucket.

        Raises:
            botocore.exceptions.ClientError: If the bucket does not exist and should_raise_head is True.

        Returns:
            dict: A dictionary with bucket information if the bucket exists.
        """
        if self.should_raise_head:
            raise botocore.exceptions.ClientError({"Error": {"Code": "404"}}, "HeadBucket")
        return {"Bucket": Bucket}

    def create_bucket(self, Bucket: str) -> None:  # noqa: N803
        """Simulate create_bucket method of S3 client.

        Args:
            Bucket (str): The name of the bucket to create.
        """
        self.created_bucket = Bucket

    def put_object(self, Body: bytes, Bucket: str, Key: str) -> None:  # noqa: N803
        """Simulate put_object method of S3 client.

        Args:
            Body (bytes): The content to upload.
            Bucket (str): The name of the bucket.
            Key (str): The key of the object.
        """
        self.put_calls.append((Body, Bucket, Key))
