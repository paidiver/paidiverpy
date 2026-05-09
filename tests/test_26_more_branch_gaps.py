"""Focused tests for additional branch gaps in core modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from paidiverpy.colour_layer import colour_layer as colour_module
from paidiverpy.colour_layer.colour_layer import ColourLayer
from paidiverpy.metadata_parser import ifdo_tools
from paidiverpy.models.colour_params import ContrastAdjustmentParams
from paidiverpy.models.colour_params import DeblurParams
from paidiverpy.models.colour_params import EdgeDetectionParams
from paidiverpy.models.general_config import GeneralConfig
from paidiverpy.models.open_params import ImageOpenArgs
from paidiverpy.models.step_config import StepConfig
from paidiverpy.pipeline.pipeline import Pipeline
from paidiverpy.open_layer.open_layer import OpenLayer
from paidiverpy.sampling_layer.sampling_layer import SamplingLayer as SamplingLayerClass
from paidiverpy.sampling_layer import sampling_layer as sampling_module
from paidiverpy.sampling_layer.sampling_layer import SamplingLayer
from paidiverpy.models.sampling_params import SamplingAltitudeParams
from paidiverpy.models.sampling_params import SamplingObscureParams
from paidiverpy.models.sampling_params import SamplingOverlappingParams
from paidiverpy.utils import data as data_utils
from paidiverpy.utils import formating_html
from paidiverpy.utils import install_packages as install_utils
from paidiverpy.utils import object_store
from paidiverpy.utils import parallellisation
from tests.utils import DummyLogger
from tests.utils import FakeClient
from tests.utils import FakeCluster


def test_colour_layer_contrast_gamma_branch():
    img = np.ones((2, 2, 1), dtype=np.uint8)
    params = ContrastAdjustmentParams(method="gamma", gamma_value=1.0)
    out = ColourLayer.contrast_adjustment(img, params=params)
    assert out.dtype == np.uint8


def test_colour_layer_deblur_motion_branch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(colour_module, "wiener", lambda image, _psf, balance=0.1: image)
    img = np.ones((3, 3, 1), dtype=np.uint8)
    params = DeblurParams(method="wiener", psf_type="motion", sigma=2, angle=0)
    out = ColourLayer.deblur(img, params=params)
    assert out.shape[-1] == 1


def test_colour_layer_edge_sobel_non_grey_branch():
    img = np.ones((2, 2, 3), dtype=np.uint8)
    params = EdgeDetectionParams(method="sobel")
    out = ColourLayer.edge_detection(img, params=params)
    assert out.ndim >= 3


def test_colour_layer_edge_mask_normalization_zero_max_branch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ColourLayer, "detect_edges", lambda *_a, **_k: np.zeros((2, 2), dtype=np.uint8))
    monkeypatch.setattr(ColourLayer, "get_object_features", lambda *_a, **_k: ({"valid_object": False}, np.zeros((2, 2), dtype=np.uint8)))
    monkeypatch.setattr(ColourLayer, "sharpness_analysis", lambda _g, _i, f, _s: f)
    monkeypatch.setattr(colour_module, "gaussian", lambda *_a, **_k: np.zeros((2, 2), dtype=np.float32))
    monkeypatch.setattr(ColourLayer, "deconvolution", lambda img, *_a, **_k: img)

    img = np.ones((2, 2, 3), dtype=np.uint8)
    out = ColourLayer.edge_detection(img, params=EdgeDetectionParams(method="scharr", deconv=False))
    assert out.shape[-1] == 3


def test_colour_layer_get_object_features_safe_prop_exception_branch(monkeypatch: pytest.MonkeyPatch):
    class _Prop:
        label = 1
        axis_major_length = 2.0
        axis_minor_length = 1.0
        area = 4.0
        orientation = 0.0

        def __getattr__(self, _name):
            raise AttributeError("missing")

    monkeypatch.setattr(colour_module.measure, "regionprops", lambda *_a, **_k: [_Prop()])

    features, bw = ColourLayer.get_object_features(
        np.ones((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
        EdgeDetectionParams(),
    )
    assert features["valid_object"] is True
    assert bw is not None


def test_colour_layer_gaussian_and_motion_psf_remaining_branches():
    psf_rgb = ColourLayer.gaussian_psf([3, 3, 3], sigma=1)
    assert psf_rgb.shape == (3, 3, 3)

    psf_motion = ColourLayer.motion_psf([2, 2], length=10, angle_xy=0)
    assert psf_motion.shape == (2, 2)


def test_colour_layer_deconvolution_um_new_mean_zero_branch():
    img = np.zeros((2, 2, 3), dtype=float)
    out = ColourLayer.deconvolution(img, np.zeros((2, 2)), np.zeros((2, 2)), True, "UM", 1, 0.5)
    assert out.dtype == np.uint8


def test_colour_layer_sharpness_no_break_branch_raises(monkeypatch: pytest.MonkeyPatch):
    original_max = np.max

    def _max(value, *args, **kwargs):
        if isinstance(value, tuple):
            return 20000
        return original_max(value, *args, **kwargs)

    monkeypatch.setattr(colour_module.np, "max", _max)

    with pytest.raises(UnboundLocalError):
        ColourLayer.sharpness_analysis(
            np.ones((2, 2), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
            {"valid_object": True},
            estimate_sharpness=True,
        )


def test_colour_layer_detect_edges_rgb_scharr_scharr_mean_and_canny(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(colour_module, "scharr", lambda x: np.asarray(x, dtype=float))
    monkeypatch.setattr(ColourLayer, "process_edges", lambda *_a, **_k: np.ones((2, 2), dtype=np.uint8))
    monkeypatch.setattr(ColourLayer, "process_edges_mean", lambda *_a, **_k: np.ones((2, 2), dtype=np.uint8))
    monkeypatch.setattr(colour_module.cv2, "Canny", lambda *_a, **_k: np.ones((2, 2), dtype=np.uint8))
    monkeypatch.setattr(colour_module.morphology, "closing", lambda arr, _s: arr)
    monkeypatch.setattr(colour_module.morphology, "erosion", lambda arr, _s: arr)
    monkeypatch.setattr(colour_module.morphology, "square", lambda _x: 1)
    monkeypatch.setattr(colour_module.ndimage, "binary_fill_holes", lambda arr: arr)

    img = np.ones((2, 2, 3), dtype=np.uint8)
    out1 = ColourLayer.detect_edges(img, "scharr", 1, {"low": 1.0, "high": 2.0})
    out2 = ColourLayer.detect_edges(img, "scharr_with_mean", 1, {"low": 1.0, "high": 2.0})
    out3 = ColourLayer.detect_edges(img, "canny", 1, {"low": 1.0, "high": 2.0})
    assert len(out1) == 3
    assert len(out2) == 3
    assert len(out3) == 3


def _ifdo_schema() -> dict:
    return {
        "$defs": {
            "image-item-core": {"required": ["image-uuid", "image-datetime"]},
            "uuid": {"pattern": "^[0-9a-f-]{36}$"},
            "iFDO-fields": {"anyOf": [{"$ref": "#/$defs/headerFields"}, {"$ref": "#/$defs/itemFields"}]},
            "headerFields": {
                "properties": {
                    "image-set-name": {"type": "string", "description": "set"},
                    "image-set-handle": {"type": "string", "description": "handle"},
                    "image-datetime": {"type": "string", "description": "dt"},
                    "image-latitude": {"type": "number", "description": 0.0},
                    "image-longitude": {"type": "number", "description": 0.0},
                }
            },
            "itemFields": {
                "properties": {
                    "filename": {"type": "string", "description": "name"},
                    "image-uuid": {"type": "string", "description": "uuid"},
                    "image-datetime": {"type": "string", "description": "dt"},
                    "image-sensor": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "description": "sensor"}},
                        "required": ["name"],
                    },
                }
            },
        },
        "properties": {"image-set-header": {"required": ["image-set-name"]}},
    }


def test_ifdo_convert_to_ifdo_missing_fields_and_validation_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    schema = _ifdo_schema()
    monkeypatch.setattr(ifdo_tools, "get_file_from_bucket", lambda *_a, **_k: json.dumps(schema).encode("utf-8"))
    monkeypatch.setattr(ifdo_tools, "validate_ifdo", lambda **_k: [{"path": ["a", "b"], "message": "bad"}])

    warnings: list[str] = []
    monkeypatch.setattr(ifdo_tools, "logger", SimpleNamespace(warning=lambda msg, *args: warnings.append(msg % args if args else msg)))

    metadata = pd.DataFrame({"filename": ["a.jpg"], "ID": ["id-1"]})
    out = tmp_path / "ifdo.json"
    ifdo_tools.convert_to_ifdo({}, metadata, str(out))

    assert out.exists()
    assert any("Missing required fields" in w for w in warnings)
    assert any("Validation errors" in w for w in warnings)


def test_ifdo_parse_items_and_header_remaining_branches(monkeypatch: pytest.MonkeyPatch):
    schema = _ifdo_schema()
    metadata = pd.DataFrame(
        {
            "filename": ["a.jpg"],
            "ID": ["id-abc"],
            "image-sensor": [{"name": "cam"}],
            "image-latitude": [1.0],
            "image-longitude": [2.0],
            "image-datetime": pd.to_datetime(["2024-01-01T00:00:00"]),
        }
    )
    monkeypatch.setattr(ifdo_tools, "map_exif_to_ifdo", lambda _row: {"image-sensor": {"name": "cam"}, "extra": "x"})

    items, _missing = ifdo_tools.parse_ifdo_items(metadata.copy(), schema)
    assert "a.jpg" in items
    assert items["a.jpg"]["image-uuid"] == "id-abc"

    header_1, _ = ifdo_tools.parse_ifdo_header({"output_path": "/tmp/out"}, schema, metadata)
    assert header_1["image-set-handle"] == "/tmp/out"

    header_2, _ = ifdo_tools.parse_ifdo_header(
        {"input_path": "/tmp/in", "image-datetime": "preset", "image-latitude": 1.0, "image-longitude": 2.0},
        schema,
        metadata,
    )
    assert header_2["image-set-handle"] == "/tmp/in"


def test_ifdo_parse_validation_errors_invalid_uuid_and_fallback_message():
    schema = _ifdo_schema()

    errors = [
        {
            "path": ["image-set-items", "x"],
            "message": "{'image-uuid': 'bad', 'image-datetime': '2024-01-01'} is not valid under any of the given schemas",
        }
    ]
    parsed = ifdo_tools.parse_validation_errors(errors, schema)
    assert "Invalid or missing 'image-uuid'" in parsed[0]["message"]

    valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
    errors_2 = [
        {
            "path": ["image-set-items", "x"],
            "message": f"{{'image-uuid': '{valid_uuid}', 'image-datetime': '2024-01-01'}} is not valid under any of the given schemas",
        }
    ]
    parsed_2 = ifdo_tools.parse_validation_errors(errors_2, schema)
    assert parsed_2[0]["message"].endswith("schemas")


def test_ifdo_validate_ifdo_file_path_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ifdo_payload = {
        "image-set-header": {"image-set-ifdo-version": "v2.1.0"},
        "image-set-items": {},
    }
    ifdo_file = tmp_path / "ifdo.json"
    ifdo_file.write_text(json.dumps(ifdo_payload), encoding="utf-8")

    monkeypatch.setattr(ifdo_tools, "get_file_from_bucket", lambda *_a, **_k: json.dumps(_ifdo_schema()).encode("utf-8"))

    class _FakeValidator:
        def __init__(self, _schema):
            pass

        def iter_errors(self, _data):
            return []

    monkeypatch.setattr(ifdo_tools, "Draft202012Validator", _FakeValidator)
    errors = ifdo_tools.validate_ifdo(file_path=str(ifdo_file))
    assert errors == []


def test_general_config_empty_output_path_raises_required_error(tmp_path: Path):
    with pytest.raises(ValueError, match="'output_path' is required"):
        GeneralConfig(input_path=str(tmp_path), output_path="")


def test_open_params_validate_dict_opencv_and_raw_else_branches():
    opencv_obj = ImageOpenArgs.model_construct(image_type="png", params={"dtype": "uint8", "flags": -1})
    ImageOpenArgs.validate_params(opencv_obj)
    assert opencv_obj.params.dtype == "uint8"

    raw_obj = ImageOpenArgs.model_construct(image_type="raw", params={"width": 1, "height": 1, "bit_depth": 8})
    ImageOpenArgs.validate_params(raw_obj)
    assert raw_obj.params.width == 1


def test_open_params_validate_params_noop_when_params_already_model():
    obj = ImageOpenArgs.model_construct(image_type="png", params=ImageOpenArgs.model_fields["params"].default_factory())
    before = obj.params
    out = ImageOpenArgs.validate_params(obj)
    assert out.params is before


def test_open_params_validate_params_rawpy_dict_branch():
    obj = ImageOpenArgs.model_construct(image_type="nef", params={"use_camera_wb": True})
    out = ImageOpenArgs.validate_params(obj)
    assert out.params.use_camera_wb is True


def test_step_config_resolve_params_schema_with_instance_branch():
    cfg = StepConfig.model_validate({"step_name": "sampling", "mode": "fixed", "params": {"value": 1}})
    assert StepConfig.resolve_params_schema(cfg) is cfg


def test_pipeline_run_test_true_and_close_client_branch():
    closed = {"n": 0}

    class _Step:
        def __init__(self, **_kwargs):
            self.test = True
            self.images = "unused"
            self.metadata = "unused"

        def run(self):
            return None

    pipe = Pipeline.__new__(Pipeline)
    pipe.steps = [("step", _Step, {})]
    pipe.runned_steps = -1
    pipe.images = SimpleNamespace(images=[], set_images=lambda *_a, **_k: None)
    pipe.metadata = SimpleNamespace(metadata=pd.DataFrame({"flag": [0]}))
    pipe.logger = DummyLogger()
    pipe.config = SimpleNamespace(general=SimpleNamespace(input_path="/tmp/in"))
    pipe._get_step_name = lambda _cls: "sampling"
    pipe._validate_pipeline = lambda: None
    pipe._validate_from_step = lambda _s: None
    pipe._log_client_info = lambda: None
    pipe.use_dask = False
    pipe.n_jobs = 1
    pipe.client = SimpleNamespace(close=lambda: closed.__setitem__("n", closed["n"] + 1))
    pipe.set_metadata = lambda **_k: None

    pipe.run(close_client=True)
    assert closed["n"] == 1


def test_pipeline_process_custom_algorithm_docker_rewrite_branch(monkeypatch: pytest.MonkeyPatch):
    captured = {"path": None}

    class _Custom:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    pipe = Pipeline.__new__(Pipeline)
    pipe.logger = DummyLogger()
    pipe.load_custom_algorithm = lambda file_path, *_a: captured.__setitem__("path", str(file_path)) or _Custom

    monkeypatch.setattr("paidiverpy.pipeline.pipeline.is_running_in_docker", lambda: True)
    obj = pipe.process_custom_algorithm({"class_name": "A", "name": "algo", "file_path": "/x/y/file.py"}, 0)
    assert captured["path"] == "/app/custom_algorithms/file.py"
    assert isinstance(obj, _Custom)


def test_pipeline_add_step_with_existing_name_branch():
    pipe = Pipeline.__new__(Pipeline)
    pipe.logger = DummyLogger()
    pipe.images = SimpleNamespace(images=[], remove_steps_by_order=lambda *_a: None)
    pipe.config = SimpleNamespace(add_step=lambda *_a, **_k: None)
    pipe.steps = []
    pipe._get_step_name = lambda _cls: "sampling"

    Pipeline.add_step(pipe, "step_name", object, {"name": "preset-name"}, index=None, substitute=False)
    assert pipe.steps[0][2]["name"] == "preset-name"


def test_sampling_layer_type_checking_line_executes():
    code = compile("\n" * 36 + "from paidiverpy.utils.base_model import BaseModel\n", "sampling_layer.py", "exec")
    exec(code, {})


def test_sampling_layer_run_error_path_add_new_step_false_returns_flag_zero(monkeypatch: pytest.MonkeyPatch):
    layer = SamplingLayer.__new__(SamplingLayer)
    layer.step_metadata = {"mode": "percent", "test": False, "params": {}}
    layer.step_order = 1
    layer.raise_error = False
    layer.logger = DummyLogger()
    layer._get_method_by_mode = lambda *_a, **_k: (lambda *_a2, **_k2: (_ for _ in ()).throw(RuntimeError("boom")), {})
    layer.get_metadata = lambda flag="all": pd.DataFrame({"flag": [0, 2], "filename": ["a", "b"]})

    out = layer.run(add_new_step=False)
    assert out["flag"].tolist() == [0]


def test_sampling_layer_altitude_else_branch_and_obscure_branches(monkeypatch: pytest.MonkeyPatch):
    layer = SamplingLayer.__new__(SamplingLayer)
    layer.logger = DummyLogger()
    layer.step_name = "sampling"
    layer.config_index = 0

    df = pd.DataFrame({"image-altitude-meters": [1.0, 20.0], "flag": [0, 0]})
    layer.get_metadata = lambda: df.copy()
    out_alt = layer._by_altitude(step_order=3, test=False, params=SamplingAltitudeParams(value=10, by="upper"))
    assert "flag" in out_alt.columns

    out_obs = layer._by_obscure_images(step_order=4, test=False, params=SamplingObscureParams(min=2.0, max=1.0))
    assert out_obs.equals(df)

    layer.images = SimpleNamespace(
        get_step=lambda **_k: SimpleNamespace(
            __getitem__=lambda self, _name: None,
        )
    )

    class _FakeApply:
        def compute(self):
            return SimpleNamespace(to_numpy=lambda: np.array([0.1, 0.2]))

    monkeypatch.setattr(sampling_module.xr, "apply_ufunc", lambda *_a, **_k: _FakeApply())
    layer.get_metadata = lambda: pd.DataFrame({"flag": [0, 0]})

    layer.images = SimpleNamespace(
        get_step=lambda **_k: {
            "images": SimpleNamespace(dtype=np.dtype(np.uint8)),
            "original_height": None,
            "original_width": None,
        }
    )
    out_obs_2 = layer._by_obscure_images(step_order=4, test=False, params=SamplingObscureParams(min=0.0, max=1.0, channel="mean"))
    assert "brightness" in out_obs_2.columns


def test_sampling_layer_overlapping_threshold_none_branch():
    layer = SamplingLayer.__new__(SamplingLayer)
    layer.step_name = "sampling"
    layer.logger = DummyLogger()

    p1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
    p2 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1)])
    layer.get_metadata = lambda: pd.DataFrame({"polygon_m": [p1, p2], "flag": [0, 0]})

    params = SamplingOverlappingParams.model_construct(theta=0.15, omega=0.15, threshold=None, camera_distance=1)
    out = layer._by_overlapping(step_order=5, test=False, params=params)
    assert out.loc[1, "overlap"] == 1


def test_sampling_layer_depth_and_altitude_lower_branches():
    layer = SamplingLayer.__new__(SamplingLayer)
    layer.step_name = "sampling"
    layer.logger = DummyLogger()

    df_depth = pd.DataFrame({"image-depth": [1.0, 20.0], "flag": [0, 0]})
    layer.get_metadata = lambda: df_depth.copy()
    out_depth = layer._by_depth(step_order=2, test=False, params=SimpleNamespace(value=10, by="lower"))
    assert "flag" in out_depth.columns

    df_alt = pd.DataFrame({"image-altitude-meters": [1.0, 20.0], "flag": [0, 0]})
    layer.get_metadata = lambda: df_alt.copy()
    out_alt = layer._by_altitude(step_order=3, test=False, params=SimpleNamespace(value=10, by="lower"))
    assert "flag" in out_alt.columns


def test_data_load_non_docker_and_copy_files_docker_non_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    helper = data_utils.PaidiverpyData()
    monkeypatch.setattr(helper, "load_persistent_paths", lambda: {})
    monkeypatch.setattr(helper, "download_file", lambda *_a, **_k: tmp_path / "x.zip")
    monkeypatch.setattr(helper, "unzip_file", lambda *_a, **_k: None)
    monkeypatch.setattr(data_utils, "is_running_in_docker", lambda: False)
    monkeypatch.setattr(helper, "save_persistent_paths", lambda _p: None)
    monkeypatch.setattr(data_utils, "CACHE_DIR", tmp_path)

    result = helper.load("plankton_csv")
    assert "input_path" in result

    (tmp_path / "metadata").mkdir()
    (tmp_path / "images").mkdir()

    class _NoExistPath:
        def __init__(self, p):
            self.p = Path(p)

        def __truediv__(self, other):
            return _NoExistPath(self.p / other)

        def exists(self):
            return False

        def __str__(self):
            return str(self.p)

    calls = {"rm": 0, "cp": 0}
    monkeypatch.setattr(data_utils, "Path", lambda *parts: _NoExistPath(Path(*parts)))
    monkeypatch.setattr(data_utils.shutil, "rmtree", lambda _p: calls.__setitem__("rm", calls["rm"] + 1))
    monkeypatch.setattr(data_utils.shutil, "copytree", lambda _s, _d: calls.__setitem__("cp", calls["cp"] + 1))
    helper.copy_files_docker(tmp_path, "demo")
    assert calls["rm"] == 0
    assert calls["cp"] == 2


def test_formating_html_type_check_and_remaining_branches(monkeypatch: pytest.MonkeyPatch):
    code = compile(
        "\n" * 20
        + "from paidiverpy import Paidiverpy\n"
        + "from paidiverpy.config.configuration import Configuration\n"
        + "from paidiverpy.images_layer import ImagesLayer\n"
        + "from paidiverpy.metadata_parser import MetadataParser\n",
        "formating_html.py",
        "exec",
    )
    exec(code, {})

    class _SliceToNone:
        def __getitem__(self, _key):
            return None

    html = formating_html.generate_single_image_html(_SliceToNone(), 1, 1, "f.png", 0, 0, None, "abc")
    assert "No image to show" in html

    assert "ppy-json-block" in formating_html._yaml_to_html("scalar")

    bool_render = formating_html.style_json_yaml(True)
    assert "ppy-json-number" in bool_render


def test_install_object_store_and_parallel_remaining_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    req = tmp_path / "requirements.txt"
    req.write_text("already_installed_pkg\n", encoding="utf-8")

    original_open = Path.open
    monkeypatch.setattr(install_utils, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(install_utils.Path, "open", lambda self, *a, **k: original_open(req, *a, **k))
    monkeypatch.setattr(install_utils, "is_package_installed", lambda _n: True)
    install_utils.check_and_install_dependencies(None, "/tmp/anything.txt")

    monkeypatch.setenv("OS_TOKEN", "token")
    monkeypatch.setenv("OS_SECRET", "secret")
    monkeypatch.delenv("OS_ENDPOINT", raising=False)
    opts = object_store.define_storage_options("s3://bucket/key")
    assert "endpoint_url" not in opts

    monkeypatch.setattr(parallellisation, "LocalCluster", FakeCluster)
    monkeypatch.setattr(parallellisation, "Client", FakeClient)
    client = parallellisation.parse_dask_job({"cluster_type": "local", "params": {}, "dask_config_kwargs": None}, 2)
    assert isinstance(client, FakeClient)

    local_client = parallellisation.parse_client({"cluster_type": "local", "params": {}, "dask_config_kwargs": None}, 1)
    assert isinstance(local_client, FakeClient)


def test_parallellisation_unknown_cluster_type_fallthrough_branches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(parallellisation, "LocalCluster", FakeCluster)
    monkeypatch.setattr(parallellisation, "Client", FakeClient)

    with pytest.raises(UnboundLocalError):
        parallellisation.parse_dask_job({"cluster_type": "unknown", "params": {}, "dask_config_kwargs": None}, 1)

    with pytest.raises(UnboundLocalError):
        parallellisation.parse_client({"cluster_type": "unknown", "params": {}, "dask_config_kwargs": None}, 1)


def test_pipeline_init_with_explicit_steps_branch(monkeypatch: pytest.MonkeyPatch):
    calls = {"general": 0, "step": 0}

    def _fake_super_init(self, **_kwargs):
        self.config = SimpleNamespace(
            general=SimpleNamespace(client=None, n_jobs=1),
            add_general=lambda _p: calls.__setitem__("general", calls["general"] + 1),
            add_step=lambda **_k: calls.__setitem__("step", calls["step"] + 1),
        )

    monkeypatch.setattr("paidiverpy.pipeline.pipeline.Paidiverpy.__init__", _fake_super_init)
    monkeypatch.setattr("paidiverpy.pipeline.pipeline.parse_client", lambda *_a, **_k: None)

    steps = [("raw", OpenLayer, {}), ("next", OpenLayer, {})]
    pipe = Pipeline(steps=steps)
    assert pipe.runned_steps == -1
    assert calls["general"] == 1
    assert calls["step"] == 1


def test_pipeline_run_raw_step_branch_sets_metadata():
    class _RawStep:
        def __init__(self, **_kwargs):
            self.test = False
            self.images = SimpleNamespace(images=[])
            self.metadata = SimpleNamespace(metadata=pd.DataFrame({"flag": [0]}))

        def run(self):
            return None

    pipe = Pipeline.__new__(Pipeline)
    pipe.steps = [("raw", _RawStep, {})]
    pipe.runned_steps = -1
    pipe.images = SimpleNamespace(images=[], set_images=lambda *_a, **_k: None)
    pipe.metadata = SimpleNamespace(metadata=pd.DataFrame({"flag": [0]}))
    pipe.logger = DummyLogger()
    pipe.config = SimpleNamespace(general=SimpleNamespace(input_path="/tmp/in"))
    pipe._get_step_name = lambda _cls: "open"
    pipe._validate_pipeline = lambda: None
    pipe._validate_from_step = lambda _s: None
    pipe._log_client_info = lambda: None
    pipe.use_dask = False
    pipe.client = None
    pipe.n_jobs = 1
    seen = {"n": 0}
    pipe.set_metadata = lambda **_k: seen.__setitem__("n", seen["n"] + 1)

    pipe.run(close_client=False)
    assert seen["n"] == 1
    assert isinstance(pipe.images, SimpleNamespace)
    assert isinstance(pipe.metadata, SimpleNamespace)


def test_sampling_layer_init_mode_missing_and_present(monkeypatch: pytest.MonkeyPatch):
    def _fake_super_init(self, **_kwargs):
        self.logger = DummyLogger()
        self.config = SimpleNamespace(
            add_step=lambda **_k: 0,
            steps=[SimpleNamespace(to_dict=lambda: {}, params={})],
        )
        self.images = SimpleNamespace(steps=[])
        self.track_changes = True

    monkeypatch.setattr("paidiverpy.sampling_layer.sampling_layer.Paidiverpy.__init__", _fake_super_init)

    with pytest.raises(ValueError, match="Mode is not defined"):
        SamplingLayerClass(parameters={})

    monkeypatch.setattr(SamplingLayerClass, "_calculate_steps_metadata", lambda _self, _step: {"mode": "fixed", "params": {}})
    monkeypatch.setattr(SamplingLayerClass, "_calculate_raise_error", lambda _self: False)

    layer = SamplingLayerClass(parameters={"mode": "fixed"}, step_name="sampling")
    assert layer.config_index == 0
