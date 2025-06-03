"""Tests for Pipeline Testing Steps."""

import unittest
from pathlib import Path
import numpy as np
from paidiverpy.colour_layer.colour_layer import ColourLayer
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from paidiverpy.sampling_layer.sampling_layer import SamplingLayer
from paidiverpy.utils.data import NUM_DIMENSIONS
from tests.base_test_class import BaseTestClass

overlapping_number_graphs = 2
obscure_number_graphs_3_channels = 4
obscure_number_graphs_1_channel = 2
resample_number_graphs = 1


class TestPipelineTestSteps(BaseTestClass):
    """Tests for Pipeline Testing Steps.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_pipeline_testing_steps(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "overlapping",
            SamplingLayer,
            {
                "mode": "overlapping",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=10)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_overlapping/graph_*.png"))
        assert len(output_graphs) == overlapping_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_overlapping/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "overlapping",
            SamplingLayer,
            {
                "mode": "overlapping",
                "params": {"theta": 40, "omega": 57, "threshold": 0.1},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        number_images += 1
        assert len(images) == number_images
        pipeline.add_step(
            "datetime",
            SamplingLayer,
            {
                "mode": "datetime",
                "test": True,
            },
        )
        assert pipeline.steps[-1][2]["test"] is True
        assert pipeline.steps[-1][0] == "datetime"
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("2_datetime/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("2_datetime/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "datetime",
            SamplingLayer,
            {
                "mode": "datetime",
                "params": {"min": "2018-06-11 04:14:00", "max": "2018-06-11 04:20:00"},
                "test": False,
            },
            2,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        assert pipeline.steps[-1][0] == "datetime"
        pipeline.run()
        images = pipeline.images.images
        number_images += 1
        assert len(images) == number_images

    def test_pipeline_testing_step_percent(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "percent",
            SamplingLayer,
            {
                "mode": "percent",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_percent/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_percent/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "percent",
            SamplingLayer,
            {
                "mode": "percent",
                "params": {"value": 0.1},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1

    def test_pipeline_testing_step_fixed(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "fixed",
            SamplingLayer,
            {
                "mode": "fixed",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_fixed/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_fixed/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "fixed",
            SamplingLayer,
            {
                "mode": "fixed",
                "test": True,
                "params": {"value": 1000},
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_fixed/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_fixed/*graph_*.png"))
        assert len(output_graphs) == 0

        pipeline.add_step(
            "fixed",
            SamplingLayer,
            {
                "mode": "fixed",
                "params": {"value": 10},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1

    def test_pipeline_testing_step_depth(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "depth",
            SamplingLayer,
            {
                "mode": "depth",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_depth/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_depth/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "depth",
            SamplingLayer,
            {
                "mode": "depth",
                "params": {"value": 10, "by": "upper"},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_depth/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_depth/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "depth",
            SamplingLayer,
            {
                "mode": "depth",
                "params": {"value": 10},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1

    def test_pipeline_testing_step_altitude(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "altitude",
            SamplingLayer,
            {
                "mode": "altitude",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_altitude/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_altitude/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "altitude",
            SamplingLayer,
            {
                "mode": "altitude",
                "params": {"value": 10},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1

    def test_pipeline_testing_step_pitch_roll(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "pitch_roll",
            SamplingLayer,
            {
                "mode": "pitch_roll",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_pitch_roll/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_pitch_roll/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "pitch_roll",
            SamplingLayer,
            {
                "mode": "pitch_roll",
                "params": {"pitch": 10, "roll": 10},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1

    def test_pipeline_testing_step_region_file(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "region",
            SamplingLayer,
            {
                "mode": "region",
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == 0
        region_filepath = Path("tests/example_files/polygons/region_polygon.geojson").absolute()
        pipeline.add_step(
            "region",
            SamplingLayer,
            {
                "mode": "region",
                "params": {"file": region_filepath},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == 0
        region_filepath = Path("tests/example_files/polygons/region_polygon.shp").absolute()
        pipeline.add_step(
            "region",
            SamplingLayer,
            {
                "mode": "region",
                "params": {"file": region_filepath},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == 0

    def test_pipeline_testing_step_region_limits(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        total_steps = 2
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "region",
            SamplingLayer,
            {
                "mode": "region",
                "params": {"limits": {"min_lon": -153.608, "max_lon": -153.605, "min_lat": 11.251, "max_lat": 11.253}},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == resample_number_graphs
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_region/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "region",
            SamplingLayer,
            {
                "mode": "region",
                "params": {"limits": {"min_lon": -153.608, "max_lon": -153.605, "min_lat": 11.251, "max_lat": 11.253}},
                "test": False,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images + 1
        assert len(pipeline.steps) == total_steps
        pipeline.add_step(
            "region",
            SamplingLayer,
            {
                "mode": "region",
                "params": {"limits": {"min_lon": -153.999, "max_lon": -153.605, "min_lat": 11.251, "max_lat": 11.253}},
                "test": False,
            },
            1,
        )
        assert pipeline.steps[-1][2]["test"] is False
        assert len(pipeline.steps) == total_steps + 1
        assert pipeline.steps[-1][2]["params"]["limits"] == {"min_lon": -153.608, "max_lon": -153.605, "min_lat": 11.251, "max_lat": 11.253}
        assert pipeline.steps[-2][2]["params"]["limits"] == {"min_lon": -153.999, "max_lon": -153.605, "min_lat": 11.251, "max_lat": 11.253}

    def test_pipeline_testing_step_obscure_three_channels(self):
        """Test the Pipeline Testing Steps."""
        number_images = 1
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert pipeline.steps[1][2]["test"]
        pipeline.add_step(
            "obscure",
            SamplingLayer,
            {
                "mode": "obscure",
                "params": {"min": 0, "max": 1, "channel": "all"},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_obscure/*graph_*.png"))
        assert len(output_graphs) == obscure_number_graphs_3_channels
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_obscure/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "obscure",
            SamplingLayer,
            {
                "mode": "obscure",
                "params": {"min": 0, "max": 1, "channel": "mean"},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_obscure/*graph_*.png"))
        assert len(output_graphs) == obscure_number_graphs_1_channel
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_obscure/*graph_*.png"))
        assert len(output_graphs) == 0
        pipeline.add_step(
            "obscure",
            SamplingLayer,
            {
                "mode": "obscure",
                "params": {"min": 0, "max": 1, "channel": "1"},
                "test": True,
            },
            1,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("1_obscure/*graph_*.png"))
        assert len(output_graphs) == obscure_number_graphs_1_channel
        for output_graph in output_graphs:
            output_graph.unlink()
        output_graphs = list(output_path.glob("1_obscure/*graph_*.png"))
        assert len(output_graphs) == 0

    def test_pipeline_testing_step_obscure_one_channel(self):
        """Test the Pipeline Testing Steps."""
        number_images = 2
        pipeline = Pipeline(config_file_path="tests/config_files/config_benthic_test_steps.yml")
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        pipeline.add_step(
            "grayscale",
            ColourLayer,
            {
                "mode": "grayscale",
            },
            1,
            substitute=True,
        )
        pipeline.run()
        images = pipeline.images.images
        assert len(images) == number_images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images[-1][0].shape) == NUM_DIMENSIONS
        pipeline.add_step(
            "obscure",
            SamplingLayer,
            {
                "mode": "obscure",
                "test": True,
            },
        )
        assert pipeline.steps[-1][2]["test"] is True
        pipeline.run(from_step=0)
        images = pipeline.images.images
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("2_obscure/*graph_*.png"))
        assert len(output_graphs) == obscure_number_graphs_1_channel
        for output_graph in output_graphs:
            output_graph.unlink()
        pipeline.add_step(
            "obscure",
            SamplingLayer,
            {
                "mode": "obscure",
                "test": False,
                "params": {"min": 0, "max": 1},
            },
            2,
            substitute=True,
        )
        assert pipeline.steps[-1][2]["test"] is False
        pipeline.run(from_step=0)
        images = pipeline.images.images
        number_images += 1
        assert len(images) == number_images
        output_path = Path(pipeline.config.general.output_path)
        output_graphs = list(output_path.glob("2_obscure/*graph_*.png"))
        assert len(output_graphs) == 0


if __name__ == "__main__":
    unittest.main()
