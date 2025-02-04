import json
import logging
import itertools
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
from paidiverpy.config.config import Configuration
from paidiverpy.pipeline.pipeline import Pipeline

def benchmark_task(configuration_file: str,
                   logger: logging.Logger
                   ):
    """Run the benchmark task.

    Args:
        configuration_file (str): The path to the configuration file.
        logger (logging.Logger): The logger to log messages.

    """
    pipeline = Pipeline(
        config_file_path=configuration_file,
        logger=logger,
        track_changes=False,
    )
    pipeline.run()

def plot_results(results: list) -> None:
    """Plot the benchmark results.

    Args:
        results (list): The list of benchmark results.
    """

    labels = [f"{r['cpus']} CPUs, {r['memory']}GB" for r in results]
    times = [r["time_taken"] for r in results]

    sorted_indices = np.argsort([r["cpus"] * 1000 + r["memory"] for r in results])
    labels = [labels[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, times, color="skyblue")

    plt.xlabel("Processing Time (seconds)")
    plt.ylabel("Configuration (CPUs, Memory)")
    plt.title("Dask Benchmark on SLURM")

    for index, value in enumerate(times):
        plt.text(value + 0.5, index, f"{value:.2f}s", va='center', fontsize=10)

    plt.gca().invert_yaxis()
    plt.savefig("benchmark_results.png")


def update_yaml(file_path, cores, processes, memory, scale, output_file):
    """Update the YAML file with new benchmarking parameters and save it."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    # Update the client params
    config["general"]["client"]["params"]["cores"] = cores
    config["general"]["client"]["params"]["processes"] = processes
    config["general"]["client"]["params"]["memory"] = f"{memory}GB"
    config["general"]["client"]["params"]["scale"] = scale

    # Save the updated config file
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_file

def benchmark_handler(benchmark_params: dict,
                      configuration_file: str,
                      logger: logging.Logger) -> None:
    """Handle the benchmark test.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.
    """
    configuration_file = Path(configuration_file)
    benchmark_results = []
    cores = benchmark_params.get("cores", [1, 2, 4, 8, 16, 32])
    processes = benchmark_params.get("processes", [1, 2, 4, 8, 16, 32])
    memory = benchmark_params.get("memory", [1, 2, 4, 8, 16, 32])
    scale = benchmark_params.get("scale", [1, 2, 4, 8, 16, 32])
    for core, proc, mem, sc in itertools.product(cores, processes, memory, scale):
        output_file = f"config_{core}_{proc}_{mem}_{sc}.yaml"

        updated_config_file = update_yaml(
            configuration_file, core, proc, mem, sc, output_file
        )

        start_time = time.perf_counter()
        benchmark_task(updated_config_file, logger)
        end_time = time.perf_counter()

        benchmark_results.append({
            "cpus": core,
            "processes": proc,
            "memory": mem,
            "scale": sc,
            "time_taken": round(end_time - start_time, 2),
        })

    print(json.dumps(benchmark_results))

    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)

    plot_results(benchmark_results)
