"""Modern web dashboard for building and running Paidiverpy pipelines."""

from __future__ import annotations


def run(host: str = "127.0.0.1", port: int = 8792, reload: bool = False) -> None:
    """Start the optional dashboard server."""
    try:
        import uvicorn  # noqa: PLC0415 - optional dependency must stay lazy
    except ImportError as exc:  # pragma: no cover - depends on optional install
        msg = "The GUI extra is not installed. Run: pip install 'paidiverpy[gui]'"
        raise RuntimeError(msg) from exc

    uvicorn.run("paidiverpy.dashboard.app:app", host=host, port=port, reload=reload)


__all__ = ["run"]
