"""FastAPI backend for the modern Paidiverpy dashboard."""

# FastAPI discovers these small route functions by their public names.
# ruff: noqa: D101, D102, D103, ANN401

from __future__ import annotations
import asyncio
import json
import subprocess
import sys
import tempfile
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any
import yaml
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware
from yaml.nodes import MappingNode
from yaml.nodes import ScalarNode
from yaml.nodes import SequenceNode
from paidiverpy.config.configuration import Configuration
from paidiverpy.dashboard.registry import dashboard_registry

DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
WORKSPACE_DIR = Path.home() / "Documents" / "paidiverpy"
CONFIG_DIR = WORKSPACE_DIR / "configs"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ConfigPayload(BaseModel):
    config: dict[str, Any]


class SavePayload(ConfigPayload):
    name: str


class YamlPayload(BaseModel):
    yaml_content: str


class ExportPayload(BaseModel):
    run_id: str
    step: int


class RunState:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.run_id: str | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=2000)
        self.condition = threading.Condition()
        self.finished = True

    def publish(self, event: dict[str, Any]) -> None:
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()


state = RunState()
runs: dict[str, dict[str, Any]] = {}
exports: dict[str, dict[str, Any]] = {}
app = FastAPI(title="Paidiverpy dashboard")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])


def _safe_name(name: str) -> str:
    clean = "".join(ch for ch in name if ch.isalnum() or ch in "-_").strip("-_")
    if not clean:
        raise HTTPException(400, "A valid configuration name is required")
    return clean


def _validate(config: dict[str, Any]) -> list[dict[str, str]]:
    try:
        Configuration.validate_config(config)
    except Exception as exc:  # noqa: BLE001
        return [{"message": str(exc)}]
    return []


def _yaml_locations(yaml_content: str) -> list[dict[str, Any]]:
    """Map YAML paths to source lines using PyYAML's composed syntax tree."""
    root = yaml.compose(yaml_content)
    if root is None:
        return []
    locations: list[dict[str, Any]] = []

    def visit(node, path: str) -> None:  # noqa: ANN001
        if path:
            locations.append(
                {
                    "path": path,
                    "start_line": node.start_mark.line,
                    "end_line": max(node.start_mark.line, node.end_mark.line - 1),
                }
            )
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                if not isinstance(key_node, ScalarNode):
                    continue
                child = f"{path}.{key_node.value}" if path else str(key_node.value)
                visit(value_node, child)
        elif isinstance(node, SequenceNode):
            for index, child_node in enumerate(node.value):
                visit(child_node, f"{path}[{index}]")

    visit(root, "")
    return locations


@app.get("/api/registry")
def registry() -> dict[str, Any]:
    return dashboard_registry()


def _raw_demo_steps() -> list[dict[str, Any]]:
    return [
        {"colour": {"name": "Grayscale", "mode": "grayscale", "params": {}}},
        {"colour": {"name": "Gaussian blur", "mode": "gaussian_blur", "params": {"sigma": 1.0}}},
        {"colour": {"name": "Contrast", "mode": "contrast", "params": {}}},
    ]


@app.get("/api/demos")
def demos() -> list[dict[str, Any]]:
    return [
        {
            "id": "quick",
            "group": "Getting started",
            "title": "Minimal starter",
            "description": "A small benthic workflow that is easy to edit and extend.",
            "dataset": "benthic_csv",
            "dataset_description": "2018 ROV benthic images from the Clarion-Clipperton Zone with CSV metadata.",
        },
        {
            "id": "benthic",
            "group": "Getting started",
            "title": "Complete benthic survey",
            "description": "Sampling, datetime selection, blur, sharpening and contrast in one workflow.",
            "dataset": "benthic_csv",
            "dataset_description": "2018 ROV benthic images from the Clarion-Clipperton Zone with CSV metadata.",
        },
        {
            "id": "plankton",
            "group": "Image types",
            "title": "Plankton restoration",
            "description": "Deblur and detect the largest object in pelagic plankton imagery.",
            "dataset": "plankton_csv",
            "dataset_description": "2023 DY157 pelagic plankton images from a vertically deployed red camera frame, with CSV metadata.",
        },
        {
            "id": "ifdo",
            "group": "Metadata and sampling",
            "title": "iFDO metadata workflow",
            "description": "Explore standards-based metadata, datetime filtering and image enhancement.",
            "dataset": "benthic_ifdo",
            "dataset_description": "2012 ROV benthic images from Haig Fras, UK, with iFDO metadata.",
        },
        {
            "id": "sampling",
            "group": "Metadata and sampling",
            "title": "Sampling strategies",
            "description": "Compare loading-time sampling with datetime selection and an overlap investigation plot.",
            "dataset": "benthic_csv",
            "dataset_description": "2018 ROV benthic images from the Clarion-Clipperton Zone with CSV metadata.",
        },
        {
            "id": "raw",
            "group": "Image types",
            "title": "Benthic raw images",
            "description": "Open Bayer raw imagery, then apply grayscale, blur and contrast.",
            "dataset": "benthic_raw_images",
            "dataset_description": "2014 ROV benthic imagery from the Clarion-Clipperton Zone, with CSV metadata.",
        },
        {
            "id": "nef",
            "group": "Image types",
            "title": "Nikon NEF images",
            "description": "Process Nikon camera raw samples using their embedded camera settings.",
            "dataset": "nef_raw",
            "dataset_description": "Sample Nikon .NEF photographs with CSV metadata.",
        },
        {
            "id": "parallel",
            "group": "Performance",
            "title": "Local parallel processing",
            "description": "Run a representative benthic workflow with two local workers.",
            "dataset": "benthic_csv",
            "dataset_description": "2018 ROV benthic images from the Clarion-Clipperton Zone with CSV metadata.",
        },
        {
            "id": "custom",
            "group": "Advanced",
            "title": "Custom MinMax algorithm",
            "description": "Apply the packaged per-image MinMaxScaler custom-layer example.",
            "dataset": "benthic_csv",
            "dataset_description": "2018 ROV benthic images from the Clarion-Clipperton Zone with CSV metadata.",
            "note": "For your own algorithm, create the Python file and CustomLayer class before replacing the example file path and class name.",
        },
    ]


@app.get("/api/demos/{demo_id}")
def demo(demo_id: str) -> dict[str, Any]:
    def general(sample: str, *, n_jobs: int = 1, initial: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        values: dict[str, Any] = {
            "sample_data": sample,
            "output_path": str(OUTPUT_DIR / demo_id),
            "track_changes": True,
            "n_jobs": n_jobs,
        }
        if initial:
            values["sampling"] = initial
        return values

    configs = {
        "benthic": {
            "general": general("benthic_csv", initial=[{"mode": "fixed", "params": {"value": 20}}]),
            "steps": [
                {"sampling": {"name": "Datetime range", "mode": "datetime", "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"}}},
                {"colour": {"name": "Gaussian blur", "mode": "gaussian_blur", "params": {"sigma": 1.0}}},
                {"colour": {"name": "Sharpen", "mode": "sharpen", "params": {"alpha": 1.5, "beta": -0.5}}},
                {"colour": {"name": "Contrast", "mode": "contrast", "params": {}}},
            ],
        },
        "plankton": {
            "general": general("plankton_csv", initial=[{"mode": "percent", "params": {"value": 0.9}}]),
            "steps": [
                {
                    "colour": {
                        "name": "Wiener deblur",
                        "mode": "deblur",
                        "params": {"method": "wiener", "psf_type": "gaussian", "sigma": 1, "angle": 45},
                    }
                },
                {
                    "colour": {
                        "name": "Edge detection",
                        "mode": "edge_detection",
                        "params": {
                            "method": "scharr",
                            "object_type": "isolated",
                            "object_selection": "largest_area",
                            "blur_radius": 3,
                            "threshold": {"low": 2.5, "high": 3.0},
                        },
                    }
                },
            ],
        },
        "quick": {
            "general": general("benthic_csv", initial=[{"mode": "fixed", "params": {"value": 8}}]),
            "steps": [
                {"colour": {"name": "Grayscale", "mode": "grayscale", "test": False, "params": {"invert_colours": False}}},
                {"colour": {"name": "Contrast", "mode": "contrast", "test": False, "params": {}}},
            ],
        },
        "ifdo": {
            "general": general("benthic_ifdo", initial=[{"mode": "fixed", "params": {"value": 5}}]),
            "steps": [
                {"colour": {"name": "Grayscale", "mode": "grayscale", "params": {}}},
                {"sampling": {"name": "Datetime range", "mode": "datetime", "params": {"min": "2016-05-11 04:14:00", "max": "2016-05-11 09:27:00"}}},
                {"colour": {"name": "Sharpen", "mode": "sharpen", "params": {"alpha": 1.5, "beta": -0.5}}},
            ],
        },
        "sampling": {
            "general": general("benthic_csv", initial=[{"mode": "percent", "params": {"value": 0.5}}]),
            "steps": [
                {"sampling": {"name": "Datetime range", "mode": "datetime", "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"}}},
                {
                    "sampling": {
                        "name": "Investigate overlap",
                        "mode": "overlapping",
                        "test": True,
                        "params": {"theta": 40, "omega": 57, "threshold": 0.5},
                    }
                },
            ],
        },
        "raw": {"general": general("benthic_raw_images"), "steps": _raw_demo_steps()},
        "nef": {"general": general("nef_raw"), "steps": _raw_demo_steps()},
        "parallel": {
            "general": general("benthic_csv", n_jobs=2, initial=[{"mode": "fixed", "params": {"value": 20}}]),
            "steps": [
                {"colour": {"name": "White balance", "mode": "colour_alteration", "params": {"method": "white_balance"}}},
                {"colour": {"name": "Gaussian blur", "mode": "gaussian_blur", "params": {"sigma": 1.0}}},
                {"colour": {"name": "Contrast", "mode": "contrast", "params": {}}},
            ],
        },
        "custom": {
            "general": general("benthic_csv", initial=[{"mode": "percent", "params": {"value": 0.1}}]),
            "steps": [
                {"colour": {"name": "Grayscale", "mode": "grayscale", "params": {}}},
                {
                    "custom": {
                        "name": "min_max_data",
                        "file_path": "example",
                        "class_name": "MyCustomClass",
                        "processing_type": "image",
                        "dependencies": "scikit-learn",
                        "params": {"feature_range": "(0, 1)"},
                    }
                },
            ],
        },
    }
    if demo_id not in configs:
        raise HTTPException(404, "Unknown demo")
    return configs[demo_id]


@app.post("/api/config/yaml")
def config_yaml(payload: ConfigPayload) -> dict[str, Any]:
    yaml_content = yaml.safe_dump(payload.config, sort_keys=False, allow_unicode=True)
    return {"yaml": yaml_content, "locations": _yaml_locations(yaml_content)}


@app.post("/api/config/parse")
def parse_yaml(payload: YamlPayload) -> dict[str, Any]:
    try:
        config = yaml.safe_load(payload.yaml_content)
    except yaml.YAMLError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not isinstance(config, dict):
        raise HTTPException(400, "The YAML document must contain a configuration mapping")
    issues = _validate(config)
    if issues:
        raise HTTPException(422, issues[0]["message"])
    return {"config": config, "locations": _yaml_locations(payload.yaml_content)}


@app.post("/api/config/validate")
def validate(payload: ConfigPayload) -> dict[str, Any]:
    issues = _validate(payload.config)
    return {"ok": not issues, "issues": issues}


@app.get("/api/configs")
def configs() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


@app.get("/api/configs/{name}")
def load_config(name: str) -> Any:
    path = CONFIG_DIR / f"{_safe_name(name)}.yaml"
    if not path.is_file():
        raise HTTPException(404, "Configuration not found")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@app.post("/api/configs")
def save_config(payload: SavePayload) -> dict[str, str]:
    issues = _validate(payload.config)
    if issues:
        raise HTTPException(422, issues[0]["message"])
    name = _safe_name(payload.name)
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload.config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"name": name, "path": str(path)}


def _read_process(process: subprocess.Popen[str], run_id: str) -> None:
    if process.stdout is None:
        state.finished = True
        state.publish({"event": "error", "run_id": run_id, "message": "The runner did not expose an output stream"})
        return
    for line in process.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"event": "log", "level": "info", "message": line.rstrip()}
        event["run_id"] = run_id
        state.publish(event)
    code = process.wait()
    state.finished = True
    state.publish({"event": "closed", "run_id": run_id, "returncode": code})


@app.post("/api/run")
def run_pipeline(payload: ConfigPayload) -> dict[str, str]:
    if state.process and state.process.poll() is None:
        raise HTTPException(409, "A pipeline is already running")
    issues = _validate(payload.config)
    if issues:
        raise HTTPException(422, issues[0]["message"])
    run_id = uuid.uuid4().hex[:12]
    run_dir = Path(tempfile.mkdtemp(prefix=f"paidiverpy-{run_id}-"))
    config_path = run_dir / "config.yaml"
    preview_path = run_dir / "previews"
    config_path.write_text(yaml.safe_dump(payload.config, sort_keys=False), encoding="utf-8")
    runs[run_id] = {"config_path": config_path, "preview_path": preview_path}
    state.events.clear()
    state.finished = False
    state.run_id = run_id
    state.process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "paidiverpy.dashboard.runner", str(config_path), str(preview_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_read_process, args=(state.process, run_id), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/previews")
def previews(run_id: str) -> dict[str, Any]:
    run = runs.get(_safe_name(run_id))
    if not run:
        raise HTTPException(404, "Run not found")
    manifest_path = run["preview_path"] / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "Previews are not ready")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for step in manifest["steps"]:
        for image in step["images"]:
            image["url"] = f"/api/runs/{run_id}/preview/{image['preview']}"
    for plot in manifest.get("investigations", []):
        plot["url"] = f"/api/runs/{run_id}/preview/{plot['preview']}"
    return manifest


@app.get("/api/runs/{run_id}/preview/{filename:path}")
def preview_file(run_id: str, filename: str) -> FileResponse:
    run = runs.get(_safe_name(run_id))
    if not run:
        raise HTTPException(404, "Run not found")
    folder = Path(run["preview_path"]).resolve()
    path = (folder / filename).resolve()
    if folder not in path.parents or not path.is_file():
        raise HTTPException(404, "Preview not found")
    return FileResponse(path)


@app.get("/api/runs/{run_id}/metadata", response_class=HTMLResponse)
def metadata_preview(run_id: str) -> HTMLResponse:
    run = runs.get(_safe_name(run_id))
    if not run:
        raise HTTPException(404, "Run not found")
    path = Path(run["preview_path"]) / "metadata.html"
    if not path.is_file():
        raise HTTPException(404, "Metadata preview is not ready")
    return HTMLResponse(path.read_text(encoding="utf-8"))


def _read_export(process: subprocess.Popen[str], export_id: str, output_path: Path) -> None:
    messages = []
    if process.stdout is not None:
        for line in process.stdout:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:  # noqa: PERF203 - runner output mixes JSON events and plain text
                messages.append({"event": "log", "message": line.rstrip()})
    returncode = process.wait()
    exports[export_id].update(
        {
            "status": "complete" if returncode == 0 else "failed",
            "output_path": str(output_path),
            "messages": messages,
        }
    )


@app.post("/api/export")
def export_images(payload: ExportPayload) -> dict[str, str]:
    run = runs.get(_safe_name(payload.run_id))
    if not run:
        raise HTTPException(404, "Run not found")
    manifest_path = run["preview_path"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"steps": []}
    available = {int(step["index"]) for step in manifest["steps"]}
    if payload.step not in available:
        raise HTTPException(400, "That step is not retained by this run")
    export_id = uuid.uuid4().hex[:12]
    output_path = OUTPUT_DIR / payload.run_id / f"step-{payload.step}"
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "paidiverpy.dashboard.runner",
            "--export",
            str(run["config_path"]),
            str(output_path),
            str(payload.step),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    exports[export_id] = {"status": "running", "output_path": str(output_path), "messages": []}
    threading.Thread(target=_read_export, args=(process, export_id, output_path), daemon=True).start()
    return {"export_id": export_id}


@app.get("/api/exports/{export_id}")
def export_status(export_id: str) -> dict[str, Any]:
    result = exports.get(_safe_name(export_id))
    if not result:
        raise HTTPException(404, "Export not found")
    return result


@app.post("/api/run/stop")
def stop_pipeline() -> dict[str, str]:
    if not state.process or state.process.poll() is not None:
        return {"status": "idle"}
    state.process.terminate()
    return {"status": "stopping"}


@app.get("/api/run/events")
async def run_events() -> StreamingResponse:
    async def stream():  # noqa: ANN202
        cursor = 0
        while True:
            events = list(state.events)
            while cursor < len(events):
                yield f"data: {json.dumps(events[cursor])}\n\n"
                cursor += 1
            if state.finished and cursor >= len(events):
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/outputs/{run_id}")
def outputs(run_id: str) -> list[dict[str, Any]]:
    safe = _safe_name(run_id)
    folder = OUTPUT_DIR / safe
    if not folder.is_dir():
        return []
    allowed = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return [
        {
            "name": path.relative_to(folder).as_posix(),
            "size": path.stat().st_size,
            "url": f"/api/outputs/{safe}/{path.relative_to(folder).as_posix()}",
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed
    ]


@app.get("/api/outputs/{run_id}/{filename:path}")
def output_file(run_id: str, filename: str) -> FileResponse:
    folder = (OUTPUT_DIR / _safe_name(run_id)).resolve()
    path = (folder / filename).resolve()
    if folder not in path.parents or not path.is_file():
        raise HTTPException(404, "Output not found")
    return FileResponse(path)


@app.exception_handler(HTTPException)
async def http_error(_request, exc: HTTPException) -> JSONResponse:  # noqa: ANN001
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
