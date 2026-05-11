"""Focused unit tests for utility helper modules."""

from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn
import pytest
from paidiverpy.utils import data as data_utils
from paidiverpy.utils import install_packages as install_utils
from paidiverpy.utils import locals as local_utils
from paidiverpy.utils import object_store
from tests.utils import DummyResponse
from tests.utils import DummyS3Client
from tests.utils import DummyTqdm


def test_data_persistence_and_calculate_information(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test the persistence of dataset paths and the calculation of dataset information.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    persistence_file = tmp_path / "datasets.json"
    monkeypatch.setattr(data_utils, "PERSISTENCE_FILE", persistence_file)

    helper = data_utils.PaidiverpyData()
    assert helper.load_persistent_paths() == {}

    helper.save_persistent_paths({"demo": str(tmp_path / "demo")})
    assert helper.load_persistent_paths() == {"demo": str(tmp_path / "demo")}

    info = helper.calculate_information(
        "dataset",
        tmp_path,
        {
            "metadata_path": "meta.csv",
            "metadata_type": "CSV_FILE",
            "image_open_args": "PNG",
            "append_data_to_metadata": True,
        },
    )
    assert info["metadata_type"] == "CSV_FILE"
    assert info["image_open_args"] == "PNG"
    assert info["append_data_to_metadata"].endswith("appended_metadata_dataset.csv")


def test_data_load_cached_and_missing_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test loading of cached dataset paths and handling of missing datasets.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    helper = data_utils.PaidiverpyData()
    cached_dir = tmp_path / "cached_ds"
    cached_dir.mkdir()

    monkeypatch.setattr(
        helper,
        "load_persistent_paths",
        lambda: {"plankton_csv": str(cached_dir)},
    )

    cached = helper.load("plankton_csv")
    assert cached["metadata_type"] == "CSV_FILE"
    assert cached["metadata_path"].endswith("metadata/metadata_plankton_csv.csv")

    monkeypatch.setattr(helper, "load_persistent_paths", dict)
    with pytest.raises(ValueError, match="Dataset 'missing_dataset' not found"):
        helper.load("missing_dataset")


def test_data_download_file_and_unzip_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test downloading and unzipping of dataset files.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    helper = data_utils.PaidiverpyData()

    monkeypatch.setattr(data_utils, "tqdm", DummyTqdm)
    monkeypatch.setattr(
        data_utils.requests,
        "get",
        lambda *args, **kwargs: DummyResponse([b"abc", b"def"]),  # noqa: ARG005
    )

    zip_path = helper.download_file("https://example.test/dataset.zip", "demo", cache_dir=tmp_path)
    assert zip_path.exists()
    assert zip_path.read_bytes() == b"abcdef"

    extract_dir = tmp_path / "extract_ok"
    archive_ok = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive_ok, "w") as archive:
        archive.writestr("images/sample.bin", b"123")
    helper.unzip_file(archive_ok, "demo", extract_dir=extract_dir)
    assert (extract_dir / "images" / "sample.bin").exists()
    assert not archive_ok.exists()

    archive_bad = tmp_path / "bad.zip"
    archive_bad.write_bytes(b"not-a-zip")
    helper.unzip_file(archive_bad, "demo", extract_dir=tmp_path / "extract_bad")
    assert not archive_bad.exists()

    helper.unzip_file(zip_path, "demo", extract_dir=extract_dir)


def test_data_copy_files_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test copying of dataset files in a Docker environment.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    helper = data_utils.PaidiverpyData()
    extract_dir = tmp_path / "dataset"
    (extract_dir / "metadata").mkdir(parents=True)
    (extract_dir / "images").mkdir(parents=True)
    (extract_dir / "metadata" / "m.txt").write_text("m", encoding="utf-8")
    (extract_dir / "images" / "i.txt").write_text("i", encoding="utf-8")

    class _FakePathObj:
        def __init__(self, path: Path):
            self.path = Path(path)

        def __truediv__(self, other: str) -> _FakePathObj:
            return _FakePathObj(self.path / other)

        def exists(self) -> bool:
            return True

        def __str__(self):
            return str(self.path)

    monkeypatch.setattr(data_utils, "Path", lambda *parts: _FakePathObj(Path(*parts)))

    calls = {"rmtree": [], "copytree": []}
    monkeypatch.setattr(data_utils.shutil, "rmtree", lambda p: calls["rmtree"].append(str(p)))
    monkeypatch.setattr(data_utils.shutil, "copytree", lambda src, dst: calls["copytree"].append((str(src), str(dst))))

    helper.copy_files_docker(extract_dir, "demo")

    assert any("/app/sample_data/demo/metadata" in p for p in calls["rmtree"])
    assert any("/app/sample_data/demo/input" in p for p in calls["rmtree"])
    assert any(dst.endswith("/app/sample_data/demo/metadata") for _, dst in calls["copytree"])
    assert any(dst.endswith("/app/sample_data/demo/input") for _, dst in calls["copytree"])


def test_object_store_paths_and_storage_options(monkeypatch: pytest.MonkeyPatch):
    """Test the definition of storage options and path handling in the object store utilities.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    monkeypatch.delenv("OS_TOKEN", raising=False)
    monkeypatch.delenv("OS_SECRET", raising=False)
    monkeypatch.delenv("OS_ENDPOINT", raising=False)

    assert object_store.define_storage_options("local/file.csv") == {}

    with pytest.raises(ValueError, match="necessary credentials"):
        object_store.define_storage_options("s3://bucket/file.csv")

    monkeypatch.setenv("OS_TOKEN", "token")
    monkeypatch.setenv("OS_SECRET", "secret")
    monkeypatch.setenv("OS_ENDPOINT", "https://s3.example")
    options = object_store.define_storage_options("s3://bucket/file.csv")
    assert options["aws_access_key_id"] == "token"
    assert options["aws_secret_access_key"] == "secret"  # noqa: S105
    assert options["endpoint_url"] == "https://s3.example"

    assert object_store.path_is_remote("https://x")
    assert object_store.path_is_remote("s3://x")
    assert not object_store.path_is_remote("./x")


def test_object_store_get_and_upload(monkeypatch: pytest.MonkeyPatch):
    """Test the retrieval and upload of files to the object store.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    s3_client = DummyS3Client()
    monkeypatch.setattr(object_store.boto3, "client", lambda **kwargs: s3_client)  # noqa: ARG005

    content = object_store.get_file_from_bucket("s3://bucket/path/to/file.bin", {"service_name": "s3"})
    assert content == b"s3-bytes"

    monkeypatch.setattr(object_store.requests, "get", lambda *args, **kwargs: DummyResponse([], content=b"http-bytes"))  # noqa: ARG005
    content_http = object_store.get_file_from_bucket("https://example.test/file.bin")
    assert content_http == b"http-bytes"

    file_obj = io.BytesIO(b"payload")
    object_store.upload_file_to_bucket(file_obj, "s3://bucket/path/to/file.bin", s3_client)
    assert s3_client.put_calls == [(b"payload", "bucket", "path/to/file.bin")]


def test_object_store_create_client_and_bucket_create(monkeypatch: pytest.MonkeyPatch):
    """Test the creation of the object store client and the handling of bucket existence.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    s3_client = DummyS3Client()
    monkeypatch.setattr(object_store, "define_storage_options", lambda _path: {"service_name": "s3"})
    monkeypatch.setattr(object_store.boto3, "client", lambda **kwargs: s3_client)  # noqa: ARG005

    client = object_store.create_client()
    assert client is s3_client

    object_store.check_create_bucket_exists("existing", s3_client)
    assert s3_client.created_bucket is None

    s3_client.should_raise_head = True
    object_store.check_create_bucket_exists("new-bucket", s3_client)
    assert s3_client.created_bucket == "new-bucket"


def test_install_packages_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test the installation helpers for packages.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    monkeypatch.setattr(install_utils, "is_package_installed", lambda _name: True)
    install_utils.check_and_install_dependencies("pytest,ruff", None)

    called = []
    monkeypatch.setattr(install_utils, "is_package_installed", lambda _name: False)
    monkeypatch.setattr(install_utils.subprocess, "check_call", lambda cmd: called.append(cmd))
    install_utils.check_and_install_dependencies("example_pkg==1.0.0", None)
    assert called

    dep_file = tmp_path / "deps.txt"
    dep_file.write_text("foo_pkg==0.1.0\n", encoding="utf-8")
    install_utils.check_and_install_dependencies(None, dep_file)

    with pytest.raises(ValueError, match="Invalid package name"):
        install_utils.check_and_install_dependencies("bad;name", None)


def test_install_packages_is_package_installed(monkeypatch: pytest.MonkeyPatch):
    """Test the function that checks if a package is installed.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    monkeypatch.setattr(install_utils, "version", lambda _name: "1.2.3")
    assert install_utils.is_package_installed("anything")

    def _raise_not_found(_name: str) -> NoReturn:
        """Simulate a package not being found by raising a PackageNotFoundError."""
        raise install_utils.PackageNotFoundError

    monkeypatch.setattr(install_utils, "version", _raise_not_found)
    assert not install_utils.is_package_installed("anything")


def test_locals_versions_helpers(monkeypatch: pytest.MonkeyPatch):
    """Test the helpers for retrieving local package and CLI versions.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    monkeypatch.setattr(local_utils, "PIP_INSTALLED", {"my-pkg": "2.0.1", "other_pkg": "3.1.4"})
    assert local_utils.pip_version("my_pkg") == "2.0.1"
    assert local_utils.pip_version("other-pkg") == "3.1.4"
    assert local_utils.pip_version("missing") == "-"

    monkeypatch.setattr(local_utils.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"tool 1.0\n"))  # noqa: ARG005
    assert local_utils.cli_version("tool") == "1.0"

    def _raise_run(*args: object, **kwargs: object) -> NoReturn:  # noqa: ARG001
        """Simulate a failure in subprocess.run by raising an exception."""
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(local_utils.subprocess, "run", _raise_run)
    monkeypatch.setattr(local_utils.shutil, "which", lambda name: "/usr/bin/tool")  # noqa: ARG005
    assert local_utils.cli_version("tool") == "- # installed"
    monkeypatch.setattr(local_utils.shutil, "which", lambda name: None)  # noqa: ARG005
    assert local_utils.cli_version("tool") == "-"


def test_locals_get_version_and_show_versions(monkeypatch: pytest.MonkeyPatch):
    """Test the functions for retrieving package versions and displaying version information.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    module_obj = SimpleNamespace(__version__="9.9.9")
    assert local_utils.get_version(module_obj) == "9.9.9"

    def _version_not_found(_name: str) -> NoReturn:
        """Simulate a package not being found by raising a PackageNotFoundError."""
        raise local_utils.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(local_utils, "version", _version_not_found)
    monkeypatch.setattr(local_utils, "pip_version", lambda _name: "1.0.0")
    assert local_utils.get_version("pkg") == "1.0.0"

    monkeypatch.setattr(local_utils, "pip_version", lambda _name: "0.0.0")
    assert local_utils.get_version("pkg") == "-"

    monkeypatch.setattr(local_utils, "get_sys_info", lambda: [("python", "3.10")])
    monkeypatch.setattr(local_utils, "get_version", lambda _name: "1.2.3")

    buffer_std = io.StringIO()
    local_utils.show_versions(file=buffer_std, conda=False)
    output_std = buffer_std.getvalue()
    assert "SYSTEM" in output_std
    assert "INSTALLED VERSIONS: CORE" in output_std

    buffer_conda = io.StringIO()
    local_utils.show_versions(file=buffer_conda, conda=True)
    output_conda = buffer_conda.getvalue()
    assert "# CORE:" in output_conda
    assert " - pandas = 1.2.3" in output_conda


def test_locals_get_sys_info_with_popen(monkeypatch: pytest.MonkeyPatch):
    """Test the retrieval of system information using subprocess.Popen.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    class _FakePath:
        """A fake Path class to simulate directory checks in get_sys_info.

        Args:
            name (str): The name of the path to simulate.
        """
        def __init__(self, name: str):
            self.name = name

        def is_dir(self) -> bool:
            """Simulate is_dir method by checking if the name is in a set of directory names."""
            return self.name in {".git", "paidiverpy"}

    class _FakePipe:
        """A fake pipe object to simulate subprocess.Popen behavior for get_sys_info."""
        returncode = 0

        def communicate(self) -> tuple[bytes, bytes]:
            """Simulate the communicate method by returning a fixed commit hash and an empty error string.

            Returns:
                tuple: A tuple containing the standard output and standard error as bytes.
            """
            return (b"deadbeef\n", b"")

    monkeypatch.setattr(local_utils, "Path", _FakePath)
    monkeypatch.setattr(local_utils.subprocess, "Popen", lambda *a, **k: _FakePipe())  # noqa: ARG005
    monkeypatch.setattr(local_utils.platform, "uname", lambda: ("Linux", "n", "6", "v", "x86_64", "proc"))

    info = dict(local_utils.get_sys_info())
    assert info["commit"] == "deadbeef"
    assert info["OS"] == "Linux"


def test_data_load_docker_branch(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Test the loading of dataset paths in a Docker environment, ensuring that provided metadata information is used for local calculations.

    Args:
        tmp_path (pytest.TempPathFactory): A temporary directory factory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    helper = data_utils.PaidiverpyData()
    dataset_name = "plankton_csv"
    dataset_info = data_utils.DATASET_URLS[dataset_name]

    extracted = tmp_path / "extract"
    extracted.mkdir()

    monkeypatch.setattr(helper, "load_persistent_paths", dict)
    monkeypatch.setattr(helper, "download_file", lambda _url, _name: tmp_path / "file.zip")
    monkeypatch.setattr(helper, "unzip_file", lambda _zip, _name, _extract: None)
    monkeypatch.setattr(data_utils, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(helper, "copy_files_docker", lambda *_args: None)

    saved = {}
    monkeypatch.setattr(helper, "save_persistent_paths", lambda paths: saved.update(paths))

    monkeypatch.setattr(data_utils, "CACHE_DIR", tmp_path)

    result = helper.load(dataset_name)
    assert result["input_path"].startswith("/app/sample_data/")
    assert saved[dataset_name].startswith("/app/sample_data/")

    local_info = helper.calculate_information(dataset_name, extracted, dataset_info)
    assert "metadata_path" in local_info


def test_data_json_roundtrip_content(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Test the roundtrip of saving and loading dataset paths in JSON format.

    Args:
        tmp_path (pytest.TempPathFactory): A temporary directory factory provided by pytest for testing file operations.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching functions and attributes during tests.
    """
    persistence_file = tmp_path / "datasets.json"
    monkeypatch.setattr(data_utils, "PERSISTENCE_FILE", persistence_file)
    helper = data_utils.PaidiverpyData()

    payload = {"a": str(tmp_path / "a"), "b": str(tmp_path / "b")}
    helper.save_persistent_paths(payload)
    assert json.loads(persistence_file.read_text(encoding="utf-8")) == payload


def test_locals_pip_version_missing_package(monkeypatch: pytest.MonkeyPatch):
    """Test pip_version when package not installed."""
    monkeypatch.setattr(local_utils, "PIP_INSTALLED", {})
    result = local_utils.pip_version("nonexistent-package")
    assert result == "-"


def test_locals_cli_version_success(monkeypatch: pytest.MonkeyPatch):
    """Test cli_version successful execution."""
    monkeypatch.setattr(
        local_utils.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=b"version 1.2.3\n"),  # noqa: ARG005
    )
    result = local_utils.cli_version("test-tool")
    assert "1.2.3" in result


def test_locals_cli_version_not_in_path(monkeypatch: pytest.MonkeyPatch):
    """Test cli_version when tool not in PATH."""
    monkeypatch.setattr(local_utils.shutil, "which", lambda name: None)  # noqa: ARG005
    result = local_utils.cli_version("missing-tool")
    assert result == "-"


def test_locals_cli_version_execution_failure(monkeypatch: pytest.MonkeyPatch):
    """Test cli_version when execution fails."""
    monkeypatch.setattr(local_utils.shutil, "which", lambda name: "/usr/bin/tool")  # noqa: ARG005

    def _raise(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        msg = "execution failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(local_utils.subprocess, "run", _raise)
    result = local_utils.cli_version("failing-tool")
    assert "- #" in result or result == "-"


def test_locals_get_version_from_module():
    """Test get_version from module __version__."""
    module = SimpleNamespace(__version__="2.0.0")
    result = local_utils.get_version(module)
    assert result == "2.0.0"


def test_locals_show_versions_conda_format(monkeypatch: pytest.MonkeyPatch):
    """Test show_versions with conda=True."""
    monkeypatch.setattr(local_utils, "get_sys_info", lambda: [("system", "test")])
    monkeypatch.setattr(local_utils, "get_version", lambda _name: "1.0.0")

    output = io.StringIO()
    local_utils.show_versions(file=output, conda=True)
    result = output.getvalue()
    assert "#" in result

def test_locals_show_versions_standard_format(monkeypatch: pytest.MonkeyPatch):
    """Test show_versions with conda=False."""
    monkeypatch.setattr(local_utils, "get_sys_info", lambda: [("python", "3.10")])
    monkeypatch.setattr(local_utils, "get_version", lambda _name: "1.0.0")

    output = io.StringIO()
    local_utils.show_versions(file=output, conda=False)
    result = output.getvalue()
    assert "SYSTEM" in result


def test_locals_get_sys_info_git_missing(monkeypatch: pytest.MonkeyPatch):
    """Test get_sys_info when .git directory missing."""
    class _FakePath:
        def __init__(self, *_args: object, **_kwargs: object):
            pass

        def is_dir(self) -> bool:
            return False

    monkeypatch.setattr(local_utils, "Path", _FakePath)
    result = dict(local_utils.get_sys_info())
    assert result.get("commit") is None
