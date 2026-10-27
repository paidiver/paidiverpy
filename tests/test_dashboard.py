"""Tests for the optional dashboard's pure API behavior."""

from paidiverpy.dashboard.registry import dashboard_registry


def test_registry_exposes_all_core_step_categories():
    """The visual step picker follows all built-in processing registries."""
    registry = dashboard_registry()
    categories = {item["name"] for item in registry["categories"]}
    assert categories == {"position", "sampling", "convert", "colour", "custom"}
    assert any(field["name"] == "sample_data" for field in registry["general"])


def test_registry_exposes_mode_defaults():
    """Pydantic defaults are retained in generated dashboard forms."""
    registry = dashboard_registry()
    colour = next(item for item in registry["categories"] if item["name"] == "colour")
    grayscale = next(item for item in colour["modes"] if item["name"] == "grayscale")
    defaults = {field["name"]: field["default"] for field in grayscale["fields"]}
    assert defaults["method"] == "opencv"
    assert defaults["invert_colours"] is False


def test_sample_data_is_a_nullable_select():
    """Users can choose a bundled sample or provide their own input paths."""
    registry = dashboard_registry()
    sample_data = next(field for field in registry["general"] if field["name"] == "sample_data")

    assert sample_data["type"] == "select"
    assert sample_data["nullable"] is True
    assert "benthic_csv" in sample_data["options"]


def test_registry_exposes_general_sampling_and_conversion():
    """Loading-time processing options are available to the visual builder."""
    registry = dashboard_registry()
    categories = {item["name"] for item in registry["general_steps"]}

    assert categories == {"sampling", "convert"}
    assert all(item["modes"] for item in registry["general_steps"])


def test_registry_exposes_custom_algorithm_fields():
    """Custom algorithms can be configured from the visual builder."""
    registry = dashboard_registry()
    custom = next(item for item in registry["categories"] if item["name"] == "custom")
    fields = {field["name"] for field in custom["modes"][0]["fields"]}

    assert {"file_path", "class_name", "processing_type", "dependencies", "params"} <= fields
