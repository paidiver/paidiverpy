"""Unit tests for the pipeline execution event contract."""

from paidiverpy.pipeline.events import RunEvent


def test_run_event_omits_empty_values() -> None:
    """Event payloads remain compact and directly JSON serializable."""
    event = RunEvent(type="step_completed", step_index=2, step_name="Resize", duration_seconds=1.25)

    assert event.to_dict() == {
        "type": "step_completed",
        "step_index": 2,
        "step_name": "Resize",
        "duration_seconds": 1.25,
    }
