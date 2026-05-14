"""Coverage-oriented unit tests for open-layer utilities, models, and parallelisation."""

from __future__ import annotations
import pytest
from paidiverpy.models.step_config import SamplingConfig
from paidiverpy.models.step_config import StepConfig
from paidiverpy.utils import parallellisation


def test_step_config_and_parallelisation(monkeypatch: pytest.MonkeyPatch):
    """Test the validation of step configuration and parallelisation."""
    step = StepConfig.model_validate({"step_name": "custom", "mode": "fixed", "params": {"raise_error": False}})
    assert step.params.raise_error is False

    with pytest.raises(ValueError, match="Unknown step_name"):
        StepConfig.model_validate({"step_name": "unknown", "mode": "fixed", "params": {}})

    with pytest.raises(ValueError, match="Missing 'mode'"):
        StepConfig.model_validate({"step_name": "sampling", "params": {}})

    with pytest.raises(ValueError, match="not valid for step"):
        StepConfig.model_validate({"step_name": "sampling", "mode": "grayscale", "params": {}})

    sampling = SamplingConfig(step_name="sampling", mode="fixed", params={"value": 2})
    sampling.update(mode="percent", params={"value": 0.2})
    assert sampling.mode == "percent"

    cpu_count = 8
    monkeypatch.setattr(parallellisation.multiprocessing, "cpu_count", lambda: cpu_count)
    assert parallellisation.get_n_jobs(-1) == cpu_count
    assert parallellisation.get_n_jobs(16) == cpu_count
    assert parallellisation.get_n_jobs(1) == 1

    slurm_cpus = 4
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", f"{slurm_cpus}")
    monkeypatch.setattr(parallellisation.multiprocessing, "cpu_count", lambda: cpu_count)
    assert parallellisation.get_n_jobs(-1) == slurm_cpus
    assert parallellisation.get_n_jobs(16) == slurm_cpus
    assert parallellisation.get_n_jobs(1) == 1

    seen = {}
    monkeypatch.setattr(parallellisation.dask.config, "set", lambda cfg: seen.update(cfg))
    parallellisation.update_dask_config({"scheduler": "threads"})
    assert seen["scheduler"] == "threads"
    parallellisation.update_dask_config(None)
