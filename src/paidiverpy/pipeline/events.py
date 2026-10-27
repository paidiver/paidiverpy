"""Structured execution events emitted by Paidiverpy pipelines."""

from __future__ import annotations
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RunEvent:
    """A lifecycle event produced while a pipeline executes."""

    type: str
    step_index: int | None = None
    step_name: str | None = None
    step_type: str | None = None
    duration_seconds: float | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without empty fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}
