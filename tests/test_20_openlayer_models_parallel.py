"""Coverage-oriented unit tests for open-layer utilities, models, and parallelisation."""

# ruff: noqa

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.models.open_params import ImageOpenArgs
from paidiverpy.models.open_params import ImageOpenArgsOpenCVParams
from paidiverpy.models.open_params import ImageOpenArgsRawParams
from paidiverpy.models.open_params import ImageOpenArgsRawPyParams
from paidiverpy.models.step_config import SamplingConfig
from paidiverpy.models.step_config import StepConfig
from paidiverpy.open_layer import utils as open_utils
from paidiverpy.utils import parallellisation
from tests.utils import FakeClient, FakeCluster




def test_open_image_remote_success_paths(monkeypatch):
    monkeypatch.setattr(open_utils, "get_file_from_bucket", lambda *_args, **_kwargs: b"bytes")
    monkeypatch.setattr(open_utils.cv2, "imdecode", lambda *_args, **_kwargs: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(open_utils, "extract_exif_single", lambda *_args, **_kwargs: {"ok": True})

    img, exif, filename = open_utils.open_image_remote(
        "s3://bucket/path/img.png",
        "png",
        {"flags": -1, "dtype": np.uint8},
    )
    assert filename == "img.png"
    assert isinstance(img, np.ndarray)
    assert exif["ok"] is True

    monkeypatch.setattr(open_utils, "load_raw_image", lambda *_args, **_kwargs: np.zeros((2, 2, 1), dtype=np.uint8))
    img_raw, _, filename_raw = open_utils.open_image_remote("s3://bucket/path/img.raw", "raw", {"bit_depth": 8})
    assert filename_raw == "img.raw"
    assert img_raw.shape[-1] == 1


def test_open_image_remote_failure_and_local(monkeypatch, tmp_path):
    monkeypatch.setattr(open_utils, "get_file_from_bucket", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")))
    img, exif, filename = open_utils.open_image_remote("s3://bucket/x.png", "png", {})
    assert img is None
    assert exif == {}
    assert filename == "x.png"

    local_img = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(open_utils, "extract_exif_single", lambda *_args, **_kwargs: {"local": True})
    monkeypatch.setattr(open_utils.cv2, "imread", lambda *_args, **_kwargs: local_img)

    img_l, exif_l, filename_l = open_utils.open_image_local(str(tmp_path / "a.png"), "png", {})
    assert filename_l == "a.png"
    assert isinstance(img_l, np.ndarray)
    assert exif_l["local"] is True


def test_correct_image_dims_and_format_branches():
    assert open_utils.correct_image_dims_and_format(None) is None

    grey = np.zeros((3, 3), dtype=np.uint8)
    out_grey_png = open_utils.correct_image_dims_and_format(grey, image_type="png")
    assert out_grey_png.shape == (3, 3, 4)

    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    out_rgb_png = open_utils.correct_image_dims_and_format(rgb, image_type="png")
    assert out_rgb_png.shape == (3, 3, 4)

    rgba_bgra = np.zeros((2, 2, 4), dtype=np.uint8)
    out_rgba = open_utils.correct_image_dims_and_format(rgba_bgra, image_type="jpg")
    assert out_rgba.shape == (2, 2, 4)

    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    out_bgr = open_utils.correct_image_dims_and_format(bgr, image_type="jpg")
    assert out_bgr.shape == (2, 2, 3)


def test_pad_and_raw_image_load_paths(monkeypatch, tmp_path):
    img = np.ones((2, 2, 3), dtype=np.uint8)
    padded = open_utils.pad_image(img, target_height=4, target_width=5)
    assert padded.shape == (4, 5, 3)

    monkeypatch.setattr(open_utils.rawpy, "LibRawFileUnsupportedError", RuntimeError)
    monkeypatch.setattr(open_utils.rawpy, "imread", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unsupported")))
    raw_fail = open_utils.load_raw_image("fake.nef", "nef", {"use_camera_wb": True}, remote=False)
    assert raw_fail is None

    monkeypatch.setattr(open_utils, "load_raw_image_using_path_open", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")))
    other_fail = open_utils.load_raw_image("fake.raw", "raw", {"bit_depth": 8}, remote=False)
    assert other_fail is None
    monkeypatch.undo()

    data = b"\x01\x02\x03\x04"
    arr_remote = open_utils.load_raw_image_using_path_open(
        BytesIO(data),
        {
            "width": 2,
            "height": 2,
            "bit_depth": 8,
            "file_header_size": 0,
            "channels": 1,
            "image_misc": "vertical_flip",
        },
        remote=True,
    )
    assert arr_remote.shape == (2, 2)

    file_path = tmp_path / "sample.raw"
    file_path.write_bytes(data)
    arr_local = open_utils.load_raw_image_using_path_open(
        file_path,
        {
            "width": 2,
            "height": 2,
            "bit_depth": 8,
            "file_header_size": 0,
            "channels": 1,
        },
        remote=False,
    )
    assert arr_local.shape == (2, 2)


def test_decode_helpers_and_exif_exceptions(monkeypatch):
    data_8 = np.arange(12, dtype=np.uint8)
    rgb = open_utils.decode_8bpp(data_8, [""], width=2, height=2, channels=3)
    assert rgb.shape == (2, 2, 3)

    packed = np.array([0xFFFF, 0x0000, 0x1F1F, 0x07E0], dtype=np.uint16)
    for layout in ["5:6:5", "5:5:5", "5:5:6", "6:5:5"]:
        decoded = open_utils.decode_16bpp(packed.copy(), layout=layout, width=2, height=2, endianess="big")
        assert decoded.shape == (2, 2, 3)

    with pytest.raises(ValueError, match="Unsupported options"):
        open_utils.decode_16bpp(packed.copy(), layout="bad-layout", width=2, height=2)

    assert open_utils.extract_exif_single("any", image_type="raw") == {}

    class _FakeImageWithExif:
        def getexif(self):
            return {1: "value"}

    monkeypatch.setattr(open_utils.Image, "open", lambda *_args, **_kwargs: _FakeImageWithExif())
    exif_ok = open_utils.extract_exif_single(BytesIO(b"x"), image_type="jpg", image_name="x.jpg")
    assert exif_ok["filename"] == "x.jpg"

    exif_no_name = open_utils.extract_exif_single(BytesIO(b"x"), image_type="jpg")
    assert exif_no_name == {}

    monkeypatch.setattr(open_utils.Image, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))
    assert open_utils.extract_exif_single("x.jpg", image_type="jpg") == {}

    monkeypatch.setattr(open_utils.Image, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad-image")))
    assert open_utils.extract_exif_single("x.jpg", image_type="jpg") == {}


def test_open_args_and_general_config_validators(monkeypatch):
    op_rawpy = ImageOpenArgs(image_type="nef", params={"use_camera_wb": True})
    assert isinstance(op_rawpy.params, ImageOpenArgsRawPyParams)

    op_cv = ImageOpenArgs(image_type="png", params={"dtype": "uint8", "flags": 1})
    assert isinstance(op_cv.params, ImageOpenArgsOpenCVParams)

    op_raw = ImageOpenArgs(image_type="raw", params={"width": 1, "height": 1, "bit_depth": 8})
    assert isinstance(op_raw.params, ImageOpenArgsRawParams)

    monkeypatch.setattr(
        "paidiverpy.models.general_config.PaidiverpyData.load",
        lambda _self, _sample: {
            "input_path": "/tmp/in",
            "metadata_path": "/tmp/meta.csv",
            "metadata_type": "CSV_FILE",
            "image_open_args": "PNG",
            "append_data_to_metadata": "/tmp/appended.csv",
        },
    )

    cfg = GeneralConfig(
        sample_data="plankton_csv",
        output_path="out",
        sampling=[{"mode": "fixed", "params": {"value": 2}}],
        convert=[{"mode": "bits", "params": {"output_bits": 8}}],
    )
    assert str(cfg.input_path) == "/tmp/in"
    assert str(cfg.metadata_path).endswith("meta.csv")
    assert cfg.sampling[0].step_name == "sampling"
    assert cfg.convert[0].step_name == "convert"

    cfg2 = GeneralConfig(input_path="/tmp/in", output_path="out")
    assert isinstance(cfg2.input_path, Path)
    assert isinstance(cfg2.output_path, Path)

    with pytest.raises(ValueError, match="Either 'sample_data' or 'input_path' must be provided"):
        GeneralConfig(output_path="out", input_path=None, sample_data=None)


def test_step_config_and_parallelisation(monkeypatch):
    step = StepConfig.model_validate({"step_name": "custom", "mode": "fixed", "params": {"raise_error": False}})
    assert step.params.raise_error is False

    with pytest.raises(ValueError, match="Unknown step_name"):
        StepConfig.model_validate({"step_name": "unknown", "mode": "fixed", "params": {}})

    with pytest.raises(ValueError, match="Missing 'mode'"):
        StepConfig.model_validate({"step_name": "sampling", "params": {}})

    with pytest.raises(ValueError, match="not valid for step"):
        StepConfig.model_validate({"step_name": "sampling", "mode": "grayscale", "params": {}})

    sampling = SamplingConfig(step_name="sampling", mode="fixed", params={"value": 2})
    sampling.update(mode="percent", params={"value": 0.2})
    assert sampling.mode == "percent"

    monkeypatch.setattr(parallellisation.multiprocessing, "cpu_count", lambda: 8)
    assert parallellisation.get_n_jobs(-1) == 8
    assert parallellisation.get_n_jobs(16) == 8
    assert parallellisation.get_n_jobs(1) == 1

    seen = {}
    monkeypatch.setattr(parallellisation.dask.config, "set", lambda cfg: seen.update(cfg))
    parallellisation.update_dask_config({"scheduler": "threads"})
    assert seen["scheduler"] == "threads"
    parallellisation.update_dask_config(None)

    monkeypatch.setattr(parallellisation, "LocalCluster", FakeCluster)
    monkeypatch.setattr(parallellisation, "SLURMCluster", FakeCluster)
    monkeypatch.setattr(parallellisation, "Client", FakeClient)

    client_local = parallellisation.parse_dask_job({"cluster_type": "local", "params": {}, "dask_config_kwargs": None}, 2)
    assert isinstance(client_local, FakeClient)

    client_slurm, job_id = parallellisation.parse_dask_job({"cluster_type": "slurm", "params": {}, "dask_config_kwargs": None}, 3)
    assert isinstance(client_slurm, FakeClient)
    assert job_id is None

    assert parallellisation.parse_client(None, 1) is None
    assert isinstance(parallellisation.parse_client({"cluster_type": "local", "params": {}, "dask_config_kwargs": None}, 2), FakeClient)
    assert isinstance(parallellisation.parse_client({"cluster_type": "slurm", "params": {}, "dask_config_kwargs": None}, 2), FakeClient)
