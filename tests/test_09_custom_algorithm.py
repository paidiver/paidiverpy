"""Tests for Pipeline with Custom Algorithm."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from paidiverpy.config.configuration import Configuration
from paidiverpy.config.configuration import GeneralConfig
from paidiverpy.pipeline import Pipeline
from tests.base_test_class import BaseTestClass


class TestPipelineCustomAlgorithm(BaseTestClass):
    """Tests for Pipeline with Custom Algorithm.

    Args:
        unittest (BaseTestClass): The unittest class.
    """

    def test_custom_algorithm(self):
        """Test generating a Pipeline with Custom Algorithm."""
        number_pipeline_steps = 3
        number_images = 3

        pipeline = Pipeline(config_file_path="tests/config_files/custom_layer/config_custom_algorithm.yml", verbose=1)

        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        assert pipeline.steps[-1][2]["step_name"] == "custom"
        assert len(pipeline.steps) == number_pipeline_steps
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images

    def test_custom_algorithm2(self):
        """Test generating a Pipeline with Custom Algorithm."""
        number_pipeline_steps = 3
        number_images = 3
        file_path = "tests/config_files/custom_layer/config_custom_algorithm2.yml"
        output_file_path = "tests/config_files/custom_layer/config_custom_algorithm2_with_dependencies_path.yml"
        requirements_path = "tests/example_files/test_requirements.txt"
        with Path(file_path).open() as file:
            data = yaml.safe_load(file)
        data["steps"][1]["custom"]["dependencies_path"] = str(Path(requirements_path).absolute())
        with Path(output_file_path).open("w") as file:
            yaml.dump(data, file, sort_keys=False)
        pipeline = Pipeline(config_file_path=output_file_path, verbose=1)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        assert pipeline.steps[-1][2]["step_name"] == "custom"
        assert len(pipeline.steps) == number_pipeline_steps
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images
        Path(output_file_path).unlink(missing_ok=True)

    def test_custom_algorithm_error(self):
        """Test generating a Pipeline with Custom Algorithm."""
        file_path = "tests/config_files/custom_layer/config_custom_algorithm2.yml"
        output_file_path = "tests/config_files/custom_layer/config_custom_algorithm2_with_error_dependency.yml"
        with Path(file_path).open() as file:
            data = yaml.safe_load(file)
        data["steps"][1]["custom"]["dependencies"] = "no-existent package ;asd1"
        with Path(output_file_path).open("w") as file:
            yaml.dump(data, file, sort_keys=False)
        with pytest.raises(ValueError) as cm:
            Pipeline(config_file_path=output_file_path, verbose=1)
        assert "Invalid package name or version" in str(cm.value)
        Path(output_file_path).unlink(missing_ok=True)

    def test_custom_algorithm_dataset(self):
        """Test generating a Pipeline with Custom Algorithm on dataset."""
        number_pipeline_steps = 3
        number_images = 3

        pipeline = Pipeline(config_file_path="tests/config_files/custom_layer/config_custom_algorithm_dataset.yml", verbose=1)

        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.config, Configuration)
        assert isinstance(pipeline.config.general, GeneralConfig)
        assert isinstance(pipeline._repr_html_(), str)
        assert isinstance(pipeline.get_metadata(), pd.DataFrame)
        assert pipeline.steps[-1][2]["step_name"] == "custom"
        assert len(pipeline.steps) == number_pipeline_steps
        pipeline.run()
        images = pipeline.images.images
        assert isinstance(images[0][0], np.ndarray)
        assert len(images) == number_images


if __name__ == "__main__":
    unittest.main()
