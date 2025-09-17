"""This module have functions related to package versions."""

import contextlib
import importlib
import json
import locale
import os
import platform
import shutil
import struct
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

PIP_INSTALLED = {}
try:
    reqs = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format", "json"])  # noqa: S603
    reqs = json.loads(reqs.decode())
    for mod in reqs:
        PIP_INSTALLED.update({mod["name"]: mod["version"]})
except subprocess.SubprocessError:
    pass


def get_sys_info() -> list[tuple[str, str]]:
    """Returns system information as a dict.

    Returns:
        list[tuple[str, str]]: A list of tuples containing system information.
    """
    blob = []

    # get full commit hash
    commit = None
    if Path(".git").is_dir() and Path("paidiverpy").is_dir():
        try:
            pipe = subprocess.Popen(  # noqa: S603
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, _ = pipe.communicate()
        except Exception:  # noqa: BLE001, S110
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                with contextlib.suppress(ValueError):
                    commit = so.decode("utf-8")
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, _, release, _, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", f"{sysname}"),
                ("OS-release", f"{release}"),
                ("machine", f"{machine}"),
                ("processor", f"{processor}"),
                ("byteorder", f"{sys.byteorder}"),
                ("LC_ALL", "{}".format(os.environ.get("LC_ALL", "None"))),
                ("LANG", "{}".format(os.environ.get("LANG", "None"))),
                ("LOCALE", "{}.{}".format(*locale.getlocale())),
            ]
        )
    except Exception:  # noqa: BLE001, S110
        pass

    return blob


def cli_version(cli_name: str) -> str:
    """Get the version of a CLI tool.

    Args:
        cli_name (str): The name of the CLI tool.

    Returns:
        str: The version of the CLI tool.
    """
    try:
        a = subprocess.run([cli_name, "--version"], capture_output=True, check=False)  # noqa: S603
        return a.stdout.decode().strip("\n").replace(cli_name, "").strip()
    except:  # noqa: E722
        if shutil.which(cli_name):
            return "- # installed"
        return "-"


def pip_version(pip_name: str) -> str:
    """Get the version of a package installed via pip.

    Args:
        pip_name (str): The name of the package.

    Returns:
        str: The version of the package.
    """
    version = "-"
    for name in [pip_name, pip_name.replace("_", "-"), pip_name.replace("-", "_")]:
        if name in PIP_INSTALLED:
            version = PIP_INSTALLED[name]
    return version


def get_version(module_name: str) -> str:
    """Get the version of a module.

    Args:
        module_name (str): The name of the module.

    Returns:
        str: The version of the module.
    """
    ver = "-"
    try:
        ver = module_name.__version__
    except AttributeError:
        try:
            ver = version(module_name)
        except importlib.metadata.PackageNotFoundError:
            try:
                ver = pip_version(module_name)
            except:  # noqa: E722
                try:  # noqa: SIM105
                    ver = cli_version(module_name)
                except:  # noqa: E722, S110
                    pass
    if sum([int(v == "0") for v in ver.split(".")]) == len(ver.split(".")):
        ver = "-"
    return ver


def show_versions(file: TextIO = sys.stdout, conda: bool = False) -> None:
    """Print the versions of paidiverpy and its dependencies.

    Args:
        file (TextIO, optional): The file to write the versions to. Defaults to sys.stdout.
        conda (bool, optional): Whether to format the output for conda. Defaults to False.
    """
    sys_info = get_sys_info()

    dependencies = {
        "core": sorted(
            [
                ("paidiverpy", get_version),
                ("pandas", get_version),
                ("pillow", get_version),
                ("scikit-image", get_version),
                ("PyYAML", get_version),
                ("opencv-python", get_version),
                ("rawpy", get_version),
                ("pydantic", get_version),
                ("scipy", get_version),
                ("xarray", get_version),
                ("openpyxl", get_version),
                ("shapely", get_version),
                ("geopandas", get_version),
                ("geopy", get_version),
                ("jsonschema", get_version),
            ]
        ),
        "ext.util": sorted(
            [
                ("tqdm", get_version),
                ("requests", get_version),
            ]
        ),
        "ext.files": sorted(
            [
                ("boto3", get_version),
                ("botocore", get_version),
            ]
        ),
        "ext.perf": sorted(
            [
                ("dask", get_version),
                ("dask-jobqueue", get_version),
                ("distributed", get_version),
            ]
        ),
        "ext.plot": sorted(
            [
                ("IPython", get_version),
                ("matplotlib", get_version),
            ]
        ),
        "dev": sorted(
            [
                ("ruff", get_version),
                ("numpy", get_version),  # will come with xarray and pandas
                ("pandas", get_version),  # will come with xarray
                ("pip", get_version),
                ("pytest", get_version),
                ("sphinx", get_version),
            ]
        ),
        "pip": sorted(
            [
                ("pytest-reportlog", get_version),
            ]
        ),
    }

    dependencies_blob = {}
    for level, deps in dependencies.items():
        deps_blob = []
        for modname, ver_f in deps:
            try:
                ver = ver_f(modname)
                deps_blob.append((modname, ver))
            except Exception:  # noqa: BLE001, PERF203
                deps_blob.append((modname, "installed"))
        dependencies_blob[level] = deps_blob

    print("\nSYSTEM", file=file)
    print("------", file=file)
    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    for level, deps_blob in dependencies_blob.items():
        if conda:
            print(f"\n# {level.upper()}:", file=file)
        else:
            title = f"INSTALLED VERSIONS: {level.upper()}"
            print(f"\n{title}", file=file)
            print("-" * len(title), file=file)
        for k, stat in deps_blob:
            if conda:
                if k != "paidiverpy":
                    kf = k.replace("_", "-")
                    comment = " " if stat != "-" else "# "
                    print(f"{comment} - {kf} = {stat}", file=file)  # Format like a conda env line, useful to update ci/requirements
            else:
                print(f"{k:<12}: {stat:<12}", file=file)
