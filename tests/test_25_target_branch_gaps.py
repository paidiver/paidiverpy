"""Focused tests for remaining branch gaps in six target modules."""

from __future__ import annotations

import importlib
import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from paidiverpy.custom_layer.custom_layer import CustomLayer
from paidiverpy.investigation_layer.investigation_layer import InvestigationLayer
from paidiverpy.open_layer.open_layer import OpenLayer
from paidiverpy.open_layer import utils as open_utils
from paidiverpy.paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.utils import locals as local_utils
from tests.utils import DummyLogger


def test_locals_import_subprocesserror_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover import-time pip list failure branch in locals module."""
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: (_ for _ in ()).throw(subprocess.SubprocessError("boom")))
    reloaded = importlib.reload(local_utils)
    assert isinstance(reloaded.PIP_INSTALLED, dict)


def test_locals_get_sys_info_exception_branches(monkeypatch: pytest.MonkeyPatch):
    """Cover get_sys_info branches for Popen failure and uname failure."""

    class _FakePath:
        def __init__(self, *_args, **_kwargs):
            pass

        def is_dir(self) -> bool:
            return True

    monkeypatch.setattr(local_utils, "Path", _FakePath)
    monkeypatch.setattr(local_utils.subprocess, "Popen", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("popen fail")))
    monkeypatch.setattr(local_utils.platform, "uname", lambda: (_ for _ in ()).throw(RuntimeError("uname fail")))

    info = dict(local_utils.get_sys_info())
    assert "commit" in info
    assert info["commit"] is None


def test_locals_get_sys_info_nonzero_git_returncode_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover get_sys_info branch where git command returns non-zero (48->54 path)."""

    class _FakePath:
        def __init__(self, *_args, **_kwargs):
            pass

        def is_dir(self) -> bool:
            return True

    class _FakePipe:
        returncode = 1

        def communicate(self):
            return (b"ignored", b"err")

    monkeypatch.setattr(local_utils, "Path", _FakePath)
    monkeypatch.setattr(local_utils.subprocess, "Popen", lambda *_a, **_k: _FakePipe())
    monkeypatch.setattr(local_utils.platform, "uname", lambda: ("Linux", "n", "6", "v", "x86_64", "proc"))

    info = dict(local_utils.get_sys_info())
    assert info["commit"] is None


def test_locals_get_version_nested_fallback_exceptions(monkeypatch: pytest.MonkeyPatch):
    """Cover get_version nested fallback except chain to final pass."""

    def _pkg_not_found(_name):
        raise local_utils.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(local_utils, "version", _pkg_not_found)
    monkeypatch.setattr(local_utils, "pip_version", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("pip fail")))
    monkeypatch.setattr(local_utils, "cli_version", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("cli fail")))

    assert local_utils.get_version("missing_mod") == "-"


def test_locals_show_versions_dependency_exception_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover show_versions branch where dependency version getter raises."""
    monkeypatch.setattr(local_utils, "get_sys_info", lambda: [("python", "3.10")])

    def _ver(name):
        if name == "paidiverpy":
            raise RuntimeError("boom")
        return "1.0.0"

    monkeypatch.setattr(local_utils, "get_version", _ver)
    out = io.StringIO()
    local_utils.show_versions(file=out, conda=False)
    text = out.getvalue()
    assert "paidiverpy" in text
    assert "installed" in text


def test_custom_layer_run_missing_method_no_raise(monkeypatch: pytest.MonkeyPatch):
    """Cover CustomLayer.run missing method branch with raise_error False."""
    obj = CustomLayer.__new__(CustomLayer)
    obj.step_metadata = {"name": "not_there", "test": False, "params": {}}
    obj.logger = DummyLogger()
    obj.raise_error = False
    obj.images = SimpleNamespace(images=[], add_step=lambda **_k: None)
    obj.track_changes = True
    obj.config_index = 1
    obj.step_name = "x"

    monkeypatch.setattr("paidiverpy.custom_layer.custom_layer.check_and_install_dependencies", lambda *_a, **_k: None)

    CustomLayer.run(obj)
    assert any(level == "error" for level, *_ in obj.logger.messages)


def test_custom_layer_run_missing_method_raises_when_configured(monkeypatch: pytest.MonkeyPatch):
    """Cover CustomLayer.run missing method branch with raise_error=True (line 101)."""
    obj = CustomLayer.__new__(CustomLayer)
    obj.step_metadata = {"name": "not_there", "test": False, "params": {}}
    obj.logger = DummyLogger()
    obj.raise_error = True
    obj.images = SimpleNamespace(images=[], add_step=lambda **_k: None)
    obj.track_changes = True
    obj.config_index = 1
    obj.step_name = "x"

    monkeypatch.setattr("paidiverpy.custom_layer.custom_layer.check_and_install_dependencies", lambda *_a, **_k: None)

    with pytest.raises(AttributeError, match="Method not_there not found"):
        CustomLayer.run(obj)


def test_custom_layer_run_test_true_skips_add_step(monkeypatch: pytest.MonkeyPatch):
    """Cover CustomLayer.run branch when test=True so add_step is skipped."""
    obj = CustomLayer.__new__(CustomLayer)
    obj.step_metadata = {"name": "algo", "test": True, "params": {}, "processing_type": "images"}
    obj.logger = DummyLogger()
    obj.raise_error = False
    obj.track_changes = True
    obj.config_index = 2
    obj.step_name = "s"
    calls: dict[str, int] = {"add": 0}
    obj.images = SimpleNamespace(images=[1], add_step=lambda **_k: calls.__setitem__("add", calls["add"] + 1))

    monkeypatch.setattr("paidiverpy.custom_layer.custom_layer.check_and_install_dependencies", lambda *_a, **_k: None)
    obj.algo = lambda *_a, **_k: None
    obj.process_images = lambda method, params: "imgs"

    CustomLayer.run(obj)
    assert calls["add"] == 0


def test_paidiverpy_run_test_branch_returns_none():
    """Cover Paidiverpy.run line 118 return path when test=True."""
    obj = Paidiverpy.__new__(Paidiverpy)
    obj.step_metadata = {"mode": "m", "test": True, "params": {}}
    obj.layer_methods = {"m": {"params": dict, "method": "dummy"}}
    obj._get_method_by_mode = lambda *_a, **_k: (lambda **_k2: None, {})
    obj.process_images = lambda *_a, **_k: xr.Dataset()
    obj.images = SimpleNamespace(add_step=lambda **_k: (_ for _ in ()).throw(AssertionError("should not add")), replace_step=lambda **_k: None)
    obj.step_name = "abc"
    obj.config_index = 0
    obj.track_changes = True

    assert obj.run(add_new_step=True) is None


def test_paidiverpy_process_images_no_images_branch(monkeypatch: pytest.MonkeyPatch):
    """Cover process_images no-images error path."""
    obj = Paidiverpy.__new__(Paidiverpy)
    obj.images = SimpleNamespace(get_step=lambda **_k: None)
    obj.logger = DummyLogger()

    monkeypatch.setattr("paidiverpy.paidiverpy.raise_value_error", lambda msg: (_ for _ in ()).throw(ValueError(msg)))

    with pytest.raises(ValueError, match="No images found to process"):
        obj.process_images(method=lambda **_k: None, params={})


def test_paidiverpy_calculate_output_image_no_zero_flag(monkeypatch: pytest.MonkeyPatch):
    """Cover calculate_output_image branch where no flag==0 exists."""
    obj = Paidiverpy.__new__(Paidiverpy)
    obj.logger = DummyLogger()

    ds = xr.Dataset(
        {
            "images": (("filename", "y", "x", "band"), np.zeros((1, 1, 1, 1), dtype=np.uint8)),
            "flag": ("filename", np.array([1], dtype=np.int64)),
            "original_height": ("filename", np.array([1], dtype=np.int64)),
            "original_width": ("filename", np.array([1], dtype=np.int64)),
            "filename": ("filename", np.array(["a.png"], dtype=object)),
        }
    )

    monkeypatch.setattr("paidiverpy.paidiverpy.raise_value_error", lambda msg: (_ for _ in ()).throw(ValueError(msg)))

    with pytest.raises(ValueError, match="No images with flag=0"):
        obj.calculate_output_image(ds, func=lambda **_k: np.zeros((1, 1, 1), dtype=np.uint8))


def test_paidiverpy_save_images_step_none_with_explicit_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover save_images branch when step is None and output_path is explicitly provided."""
    obj = Paidiverpy.__new__(Paidiverpy)
    obj.config = SimpleNamespace(general=SimpleNamespace(output_path=tmp_path / "default"))
    obj.logger = DummyLogger()
    seen: dict[str, object] = {}
    obj.images = SimpleNamespace(save=lambda **kwargs: seen.update(kwargs))
    obj.set_metadata = lambda **kwargs: seen.update({"dataset_metadata": kwargs.get("dataset_metadata")})
    obj.client = None
    obj.n_jobs = 1
    obj.use_dask = False

    out = tmp_path / "explicit"
    obj.save_images(step=None, output_path=out)
    assert seen["last"] is True
    assert seen["output_path"] == out


def test_paidiverpy_save_images_step_not_none_branch(tmp_path: Path):
    """Cover save_images branch 329->331 by taking step-is-not-None path."""
    obj = Paidiverpy.__new__(Paidiverpy)
    obj.config = SimpleNamespace(general=SimpleNamespace(output_path=tmp_path / "default"))
    obj.logger = DummyLogger()
    seen: dict[str, object] = {}
    obj.images = SimpleNamespace(save=lambda **kwargs: seen.update(kwargs))
    obj.set_metadata = lambda **kwargs: seen.update({"dataset_metadata": kwargs.get("dataset_metadata")})
    obj.client = None
    obj.n_jobs = 1
    obj.use_dask = False

    out = tmp_path / "explicit2"
    obj.save_images(step=1, output_path=out)
    assert seen["last"] is False
    assert seen["step"] == 1


def test_paidiverpy_get_method_by_mode_cast_params_branch():
    """Cover _get_method_by_mode params casting path."""

    class _Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    obj = Paidiverpy.__new__(Paidiverpy)
    obj.foo = lambda *_a, **_k: None
    method, params = obj._get_method_by_mode({"a": 1}, {"m": {"params": _Params, "method": "foo"}}, "m", class_method=False)
    assert callable(method)
    assert isinstance(params, _Params)


def test_paidiverpy_calculate_raise_error_dict_branch():
    """Cover _calculate_raise_error branch when params is a dict."""
    obj = Paidiverpy.__new__(Paidiverpy)
    obj.raise_error = False
    obj.step_metadata = {"params": {"raise_error": True}}
    assert obj._calculate_raise_error() is True


def test_paidiverpy_process_single_output_band_mismatch_branch():
    """Cover process_single early return when output bands mismatch."""

    def _func(**_k):
        return np.zeros((2, 2, 1), dtype=np.uint8)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out, h, w = Paidiverpy.process_single(
        img=img,
        flag=0,
        height=2,
        width=2,
        filename="a.png",
        output_bands=2,
        func=_func,
        metadata=pd.DataFrame(),
    )
    assert out.shape == (2, 2, 2)
    assert h == 2
    assert w == 2


def test_open_layer_process_single_image_returns_none_when_image_missing():
    """Cover OpenLayer.process_single_image branch where loader returns None image."""

    def _func(**_k):
        return None, {}, "x.png"

    result = OpenLayer.process_single_image(
        img_path="/tmp/x.png",
        func=_func,
        metadata=pd.Series({"image-datetime": pd.Timestamp("2024-01-01")}),
        rename="datetime",
        image_type="png",
        image_open_args={},
        storage_options={},
    )
    assert result is None


def test_open_layer_get_image_open_args_configparams_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cover _get_image_open_args branch for ConfigParams instance input."""
    obj = OpenLayer.__new__(OpenLayer)

    cfg = ConfigParams.model_construct(
        input_path=tmp_path,
        output_path=tmp_path,
        metadata_path=tmp_path / "meta.csv",
        metadata_type="CSV_FILE",
        image_open_args={"image_type": "png", "params": {"flags": -1, "dtype": np.uint8}},
        track_changes=True,
        n_jobs=1,
    )
    monkeypatch.setattr(
        ConfigParams,
        "to_dict",
        lambda _self: {"image_type": "png", "params": {"flags": -1, "dtype": np.uint8}},
    )

    image_type, args = obj._get_image_open_args(cfg)
    assert image_type == "png"
    assert args["flags"] == -1


def test_open_utils_correct_image_dims_png_extra_channel_hits_alpha_append():
    """Cover line where PNG path appends alpha channel via dstack."""
    img = np.zeros((2, 2, 2), dtype=np.uint8)
    out = open_utils.correct_image_dims_and_format(img, image_type="png")
    assert out.shape[-1] == 3


def test_open_utils_pad_image_2d_branch():
    """Cover pad_image branch where no channel dimension padding is added."""
    img = np.ones((2, 2), dtype=np.uint8)
    out = open_utils.pad_image(img, target_height=3, target_width=4)
    assert out.shape == (3, 4)


def test_open_utils_load_raw_image_using_path_open_16bit_and_invalid_depth(tmp_path: Path):
    """Cover 16-bit decode branch and unsupported bit-depth ValueError branch."""
    sixteen_file = tmp_path / "sample16.raw"
    sixteen_file.write_bytes((np.array([0x07E0], dtype=np.uint16)).tobytes())

    out = open_utils.load_raw_image_using_path_open(
        sixteen_file,
        {
            "width": 1,
            "height": 1,
            "bit_depth": 16,
            "endianness": "little",
            "layout": "5:6:5",
            "image_misc": "",
            "file_header_size": 0,
            "channels": 1,
        },
        remote=False,
    )
    assert out.shape == (1, 1, 3)

    with pytest.raises(ValueError, match="Unsupported bit depth"):
        open_utils.load_raw_image_using_path_open(
            sixteen_file,
            {
                "width": 1,
                "height": 1,
                "bit_depth": 12,
                "file_header_size": 0,
                "channels": 1,
            },
            remote=False,
        )


def test_investigation_layer_run_remote_skips_processing():
    """Cover InvestigationLayer.run remote early-return branch."""
    obj = InvestigationLayer.__new__(InvestigationLayer)
    obj.logger = DummyLogger()
    obj.is_remote = True
    obj.output_path = Path("/tmp")
    obj.step_order = 1
    obj.step_name = "s"
    obj.plot_metadata = pd.DataFrame()
    obj.plots = []

    InvestigationLayer.run(obj)
    assert any(level == "error" for level, *_ in obj.logger.messages)


def test_investigation_layer_run_sets_plot_metadata_when_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover run branch where plot_metadata is fetched via get_metadata."""
    obj = InvestigationLayer.__new__(InvestigationLayer)
    obj.logger = DummyLogger()
    obj.is_remote = False
    obj.output_path = tmp_path
    obj.step_order = 2
    obj.step_name = "invest"
    obj.plot_metadata = None
    obj.plots = []
    obj.get_metadata = lambda: pd.DataFrame({"flag": [0]})

    InvestigationLayer.run(obj)
    assert isinstance(obj.plot_metadata, pd.DataFrame)


def test_investigation_layer_plot_trimmed_photos_missing_columns_warns():
    """Cover missing longitude/latitude branch in plot_trimmed_photos."""
    obj = InvestigationLayer.__new__(InvestigationLayer)
    obj.logger = DummyLogger()
    obj.get_metadata = lambda: pd.DataFrame({"flag": [0]})
    obj.metadata = SimpleNamespace(dataset_metadata={})
    obj.output_path = Path("/tmp")

    obj.plot_trimmed_photos(pd.DataFrame({"flag": [0]}))
    warnings = [m for m in obj.logger.messages if m[0] == "warning"]
    assert len(warnings) >= 1


def test_investigation_layer_plot_polygons_no_legend_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover plot_polygons branch where both overlap groups are empty (no legend call)."""
    obj = InvestigationLayer.__new__(InvestigationLayer)
    obj.output_path = tmp_path
    obj.plot_metadata = pd.DataFrame({"overlap": [], "polygon_m": []})

    legend_calls = {"n": 0}
    monkeypatch.setattr("paidiverpy.investigation_layer.investigation_layer.plt.legend", lambda: legend_calls.__setitem__("n", legend_calls["n"] + 1))

    obj.plot_polygons()
    assert legend_calls["n"] == 0


def test_investigation_layer_plot_brightness_hist_no_columns_warns():
    """Cover plot_brightness_hist branch with no brightness columns."""
    obj = InvestigationLayer.__new__(InvestigationLayer)
    obj.logger = DummyLogger()
    obj.output_path = Path("/tmp")

    obj.plot_brightness_hist(pd.DataFrame({"filename": ["a.png"]}))
    assert any(level == "warning" for level, *_ in obj.logger.messages)
