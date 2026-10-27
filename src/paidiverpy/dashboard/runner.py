"""Subprocess entry point used by the dashboard."""

# The runner protocol deliberately writes newline-delimited JSON to stdout.
# ruff: noqa: D102, D103, T201, TRY300

from __future__ import annotations
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
import numpy as np
import psutil
from paidiverpy.pipeline import Pipeline
from paidiverpy.pipeline import RunEvent
from paidiverpy.utils.formating_html import numpy_array_to_image_bytes

_EMIT_LOCK = threading.Lock()


class EventHandler(logging.Handler):
    """Forward regular Paidiverpy logs as newline-delimited dashboard events."""

    def emit(self, record: logging.LogRecord) -> None:
        emit("log", level=record.levelname.lower(), message=self.format(record))


def emit(event: str, **payload: object) -> None:
    with _EMIT_LOCK:
        print(json.dumps({"event": event, **payload}, default=str), flush=True)


class MemoryMonitor:
    """Sample runner and Dask child-process memory without changing core Paidiverpy."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.started = time.monotonic()
        self.run_peak = 0.0
        self.step_start = 0.0
        self.step_peak = 0.0
        self.step_index: int | None = None
        self.step_name = "startup"
        self.pipeline: Pipeline | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def rss_mb(self) -> float:
        """Return aggregate RSS for the runner and all living descendants."""
        processes = [self.process, *self.process.children(recursive=True)]
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # noqa: PERF203 - child processes may exit during sampling
                continue
        return total / 1024**2

    def sample(self) -> float:
        """Record and return the latest aggregate memory reading."""
        rss = self.rss_mb()
        with self.lock:
            self.run_peak = max(self.run_peak, rss)
            if self.step_index is not None:
                self.step_peak = max(self.step_peak, rss)
        return rss

    def loop(self) -> None:
        """Emit periodic samples until pipeline processing finishes."""
        while not self.stop_event.wait(0.5):
            rss = self.sample()
            emit(
                "memory_sample",
                current_mb=round(rss, 1),
                run_peak_mb=round(self.run_peak, 1),
                elapsed_seconds=round(time.monotonic() - self.started, 2),
            )

    def dataset_mb(self) -> float | None:
        """Estimate retained xarray storage without computing lazy arrays."""
        if self.pipeline is None:
            return None
        try:
            return self.pipeline.images.images.nbytes / 1024**2
        except (AttributeError, TypeError):
            return None

    def handle(self, event: RunEvent) -> None:
        """Forward core lifecycle events and attach step memory summaries."""
        payload = event.to_dict()
        emit("run_event", **payload)
        if event.type == "step_started":
            rss = self.sample()
            with self.lock:
                self.step_index = event.step_index
                self.step_name = event.step_name or f"Step {event.step_index}"
                self.step_start = rss
                self.step_peak = rss
        elif event.type in {"step_completed", "step_failed"}:
            rss = self.sample()
            dataset_mb = self.dataset_mb()
            with self.lock:
                step_peak = self.step_peak
                step_start = self.step_start
                self.step_index = None
            emit(
                "memory_step",
                step_index=event.step_index,
                step_name=event.step_name,
                current_mb=round(rss, 1),
                step_start_mb=round(step_start, 1),
                step_peak_mb=round(step_peak, 1),
                run_peak_mb=round(self.run_peak, 1),
                dataset_mb=round(dataset_mb, 1) if dataset_mb is not None else None,
                elapsed_seconds=round(time.monotonic() - self.started, 2),
            )


def _collect_investigation_plots(pipeline: Pipeline, preview_dir: Path, started_at: float) -> list[dict[str, object]]:
    output_path, is_remote = pipeline.config.get_output_path()
    if is_remote:
        return []
    plot_dir = preview_dir / "investigations"
    plots = []
    titles = {
        "graph_trimmed_images": "Original vs sampled locations",
        "graph_polygons": "Image polygon overlap",
    }
    for path in sorted(Path(output_path).rglob("graph_*.png")):
        if path.stat().st_mtime < started_at - 1:
            continue
        plot_dir.mkdir(parents=True, exist_ok=True)
        preview_name = f"{len(plots)}-{path.name}"
        shutil.copy2(path, plot_dir / preview_name)
        title = titles.get(path.stem, path.stem.removeprefix("graph_").replace("_", " ").title())
        plots.append(
            {
                "index": len(plots),
                "step": path.parent.name,
                "title": title,
                "filename": path.name,
                "preview": f"investigations/{preview_name}",
            }
        )
    return plots


def _write_previews(pipeline: Pipeline, preview_dir: Path, started_at: float, limit: int = 12) -> dict[str, object]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    retained = {name for name in pipeline.images.images.data_vars if name.startswith("images_")}
    retained_indices = [index for index, _name in enumerate(pipeline.images.steps) if f"images_{index}" in retained]
    final_filenames: list[str] = []
    if retained_indices:
        final_index = retained_indices[-1]
        final_dataset = pipeline.images.get_step(step=final_index)
        if "flag" in pipeline.images.images.coords:
            flags = pipeline.images.images["flag"].values
            final_dataset = final_dataset.isel(filename=np.where((flags == 0) | (flags > final_index))[0])
        final_filenames = [str(value) for value in final_dataset["filename"].values[:limit]]
    steps = []
    for index, name in enumerate(pipeline.images.steps):
        if f"images_{index}" not in retained:
            continue
        dataset = pipeline.images.get_step(step=index)
        if "flag" in pipeline.images.images.coords:
            flags = pipeline.images.images["flag"].values
            keep = np.where((flags == 0) | (flags > index))[0]
            dataset = dataset.isel(filename=keep)
        count = int(dataset.sizes.get("filename", 0))
        step_dir = preview_dir / f"step-{index}"
        step_dir.mkdir(parents=True, exist_ok=True)
        images = []
        filename_indices = {str(value): position for position, value in enumerate(dataset["filename"].values)}
        selected_indices = [filename_indices[filename] for filename in final_filenames if filename in filename_indices]
        for image_index in selected_indices:
            height = int(dataset["original_height"].isel(filename=image_index).item())
            width = int(dataset["original_width"].isel(filename=image_index).item())
            filename = str(dataset["filename"].isel(filename=image_index).item())
            array = dataset["images"].isel(filename=image_index).data[:height, :width, :]
            image_bytes, image_format = numpy_array_to_image_bytes(array, (640, 640))
            preview_name = f"{len(images)}.{image_format.lower()}"
            (step_dir / preview_name).write_bytes(image_bytes)
            images.append(
                {
                    "index": image_index,
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "preview": f"step-{index}/{preview_name}",
                }
            )
        steps.append({"index": index, "name": name, "count": count, "images": images})
    metadata = pipeline.get_metadata()
    if hasattr(metadata, "compute"):
        metadata = metadata.compute()
    metadata_preview = metadata.head(100)
    metadata_html = metadata_preview.to_html(
        index=False,
        border=0,
        classes=["metadata-dataframe"],
        escape=True,
        justify="left",
    )
    (preview_dir / "metadata.html").write_text(metadata_html, encoding="utf-8")
    manifest = {
        "track_changes": pipeline.track_changes,
        "available_images": final_filenames,
        "steps": steps,
        "metadata": {
            "count": len(metadata),
            "displayed": len(metadata_preview),
            "columns": len(metadata.columns),
        },
        "investigations": _collect_investigation_plots(pipeline, preview_dir, started_at),
    }
    (preview_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    export_mode = len(sys.argv) > 1 and sys.argv[1] == "--export"
    offset = 2 if export_mode else 1
    config_path = Path(sys.argv[offset]).resolve()
    target_path = Path(sys.argv[offset + 1]).resolve()
    logger = logging.getLogger("paidiverpy")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(EventHandler())
    monitor = MemoryMonitor()
    sampler = threading.Thread(target=monitor.loop, daemon=True)
    sampler.start()
    started_at = time.time()
    try:
        emit("status", state="running", message="Loading pipeline")
        pipeline = Pipeline(config_file_path=str(config_path), logger=logger, event_callback=monitor.handle)
        monitor.pipeline = pipeline
        emit("status", state="running", message="Processing images")
        pipeline.run(close_client=False, save_images=False)
        if export_mode:
            step = int(sys.argv[offset + 2])
            target_path.mkdir(parents=True, exist_ok=True)
            pipeline.save_images(step=step, output_path=str(target_path))
            emit("export_complete", step=step, output_path=str(target_path))
        else:
            manifest = _write_previews(pipeline, target_path, started_at)
            emit("previews_ready", manifest=manifest)
            emit("complete", count=sum(int(step["count"]) for step in manifest["steps"]))
        if pipeline.client:
            pipeline.client.close()
        return 0
    except Exception as exc:  # noqa: BLE001
        emit("error", message=str(exc), error_type=type(exc).__name__)
        return 1
    finally:
        monitor.stop_event.set()
        sampler.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
