"""Tests for benchmark batch submission helpers."""

from __future__ import annotations
import logging
from pathlib import Path

import pytest
from paidiverpy.utils.benchmark import benchmark_test


def test_build_benchmark_sbatch_script(monkeypatch: pytest.MonkeyPatch):
    """Test generating a benchmark sbatch wrapper."""
    monkeypatch.setenv("PAYDIVERPY_ENV", "PaidiverpyNew")
    monkeypatch.setattr(benchmark_test.shutil, "which", lambda command: "/usr/bin/paidiverpy" if command == "paidiverpy" else None)

    script = benchmark_test.build_benchmark_sbatch_script(
        "/tmp/config.yml",
        {
            "cluster_type": "slurm",
            "cores": [4, 12],
            "memory": [32],
            "walltime": "01:23:45",
            "queue": "standard",
            "account": "paidiver",
            "n_jobs": [1, 2],
        },
    )

    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=32GB" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --account=paidiver" in script
    assert "micromamba activate PaidiverpyNew" in script or "conda activate PaidiverpyNew" in script
    assert '"cluster_type": "threads"' in script
    assert '"results_cluster_type": "slurm"' in script
    assert "paidiverpy -bt" in script


def test_benchmark_handler_slurm_submits_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: str):
    """Test that Slurm benchmark mode submits a batch job instead of running inline."""
    submitted = {}

    def fake_submit(configuration_file: str, benchmark_params: dict, logger: logging.Logger) -> None:
        submitted["configuration_file"] = configuration_file
        submitted["benchmark_params"] = benchmark_params
        submitted["logger_name"] = logger.name

    monkeypatch.setattr(benchmark_test, "submit_benchmark_sbatch", fake_submit)

    benchmark_test.benchmark_handler(
        {
            "cluster_type": "slurm",
            "cores": [16],
            "memory": [64],
            "walltime": "12:00:00",
            "queue": "standard",
            "n_jobs": [1, 2],
        },
        str(Path(tmp_path) / "config.yml"),
        logging.getLogger("test-benchmark"),
    )

    assert submitted["configuration_file"] == str(Path(tmp_path) / "config.yml")
    assert submitted["benchmark_params"]["cluster_type"] == "slurm"
