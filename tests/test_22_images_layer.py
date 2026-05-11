"""Focused tests for remaining branch gaps in images_layer and configuration."""

from __future__ import annotations
from types import SimpleNamespace
from typing import TYPE_CHECKING
import numpy as np
import pytest
import xarray as xr
from paidiverpy.images_layer import ImagesLayer

if TYPE_CHECKING:
    import io
    from pathlib import Path

def test_images_layer_save_remote_creates_client_and_bucket(monkeypatch: pytest.MonkeyPatch):
    """Cover remote-save branch that creates client and ensures bucket exists."""
    images_layer = ImagesLayer(output_path="unused")

    fake_images = xr.Dataset(
        {
            "images": (("filename", "y", "x", "band"), np.zeros((1, 1, 1, 1), dtype=np.uint8)),
            "original_height": ("filename", np.array([1])),
            "original_width": ("filename", np.array([1])),
        },
        coords={"filename": np.array(["a.png"], dtype=object)},
    )
    monkeypatch.setattr(images_layer, "get_step", lambda *_a, **_k: fake_images)

    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "paidiverpy.images_layer.create_client",
        lambda: "fake-client",
    )
    monkeypatch.setattr(
        "paidiverpy.images_layer.check_create_bucket_exists",
        lambda bucket, client: seen.update({"bucket": bucket, "client": client}),
    )

    monkeypatch.setattr(
        "paidiverpy.images_layer.xr.apply_ufunc",
        lambda *_a, **_k: SimpleNamespace(compute=lambda: None),
    )

    cfg = SimpleNamespace(get_output_path=lambda _out: ("s3://my-bucket/out/", True))
    images_layer.save(config=cfg, use_dask=False)

    assert seen["bucket"] == "my-bucket"
    assert seen["client"] == "fake-client"


def test_images_layer_process_and_upload_uint16_remote_upload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover uint16+tiff/png+s3 branch in process_and_upload."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2), dtype=np.uint16)

    uploaded: dict[str, object] = {}

    class _FakePILImage:
        def save(self, buffer: io.BytesIO, format: str) -> None:  # noqa: A002, ARG002
            buffer.write(b"img")

    monkeypatch.setattr("paidiverpy.images_layer.Image.fromarray", lambda _arr: _FakePILImage())
    monkeypatch.setattr(
        "paidiverpy.images_layer.upload_file_to_bucket",
        lambda buffer, path, client: uploaded.update({"size": len(buffer.getvalue()), "path": path, "client": client}),
    )

    images_layer.process_and_upload(image, tmp_path / "out", "png", s3_client="s3")

    assert uploaded["size"] > 0
    assert str(uploaded["path"]).endswith(".png")
    assert uploaded["client"] == "s3"


def test_images_layer_process_and_upload_uint16_invalid_format_raises(tmp_path: Path):
    """Cover uint16 invalid-format ValueError branch."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2), dtype=np.uint16)

    with pytest.raises(ValueError, match="16-bit images can only be saved"):
        images_layer.process_and_upload(image, tmp_path / "out", "jpg", s3_client=None)


def test_images_layer_process_and_upload_uint8_remote_upload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover uint8/float32+s3 branch in process_and_upload."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "paidiverpy.images_layer.cv2.imencode",
        lambda *_a, **_k: (True, np.array([1, 2, 3], dtype=np.uint8)),
    )

    uploaded_size = 3

    uploaded: dict[str, object] = {}
    monkeypatch.setattr(
        "paidiverpy.images_layer.upload_file_to_bucket",
        lambda buffer, path, client: uploaded.update({"size": len(buffer.getvalue()), "path": path, "client": client}),
    )

    images_layer.process_and_upload(image, tmp_path / "out", "png", s3_client="s3")

    assert uploaded["size"] == uploaded_size
    assert str(uploaded["path"]).endswith(".png")
    assert uploaded["client"] == "s3"


def test_images_layer_process_and_upload_unsupported_dtype_raises(tmp_path: Path):
    """Cover unsupported-dtype ValueError branch."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2, 3), dtype=np.int64)

    with pytest.raises(ValueError, match="Unsupported image dtype"):
        images_layer.process_and_upload(image, tmp_path / "out", "png", s3_client=None)


def test_images_layer_calculate_image_single_channel_squeeze_branch():
    """Cover calculate_image branch for trailing single channel arrays."""
    images_layer = ImagesLayer(output_path="unused")
    image = np.ones((2, 2, 1), dtype=np.uint8)

    out = images_layer.calculate_image(image)
    assert out.shape == (2, 2)


def test_images_layer_remove_nonexistent_path_and_call_with_explicit_max(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cover remove() no-op when path does not exist and __call__ explicit max_images branch."""
    images_layer = ImagesLayer(output_path=tmp_path / "missing-dir")

    monkeypatch.setattr("paidiverpy.images_layer.is_running_in_docker", lambda: False)
    images_layer.remove()

    called: dict[str, object] = {}
    monkeypatch.setattr(
        "paidiverpy.images_layer.formating_html.images_repr",
        lambda _layer, max_images=None, html=False, **_k: called.update({"max_images": max_images, "html": html}) or "ok",
    )
    max_images = 5
    rendered = images_layer(max_images=max_images)
    assert rendered == "ok"
    assert called["max_images"] == max_images
    assert called["html"] is True

    called.clear()
    rendered_default = images_layer()
    assert rendered_default == "ok"
    assert called["max_images"] == images_layer.max_images
    assert called["html"] is True
