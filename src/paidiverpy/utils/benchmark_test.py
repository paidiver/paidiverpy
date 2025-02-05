import gc
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
    del pipeline
    gc.collect()

def plot_results(results: list, cluster_type: str) -> None:
    """Plot the benchmark results.

    Args:
        results (list): The list of benchmark results.
        cluster_type (str): The cluster type.
    """

    if cluster_type == "local":
        labels = [f"{r['workers']} Workers, {r['threads']} Threads, {r['memory']}GB" for r in results]
        sorted_indices = np.argsort([r["workers"] * 1000 + r["threads"] * 100 + r["memory"] for r in results])
        y_label = "Configuration (Workers, Threads, Memory)"
    else:
        labels = [f"{r['cpus']} CPUs, {r['memory']}GB" for r in results]
        sorted_indices = np.argsort([r["cpus"] * 1000 + r["memory"] for r in results])
        y_label = "Configuration (CPUs, Memory)"

    times = [r["time_taken"] for r in results]

    labels = [labels[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, times, color="skyblue")

    plt.xlabel("Processing Time (seconds)")
    plt.ylabel(y_label)
    plt.title(f"Dask Benchmark on {cluster_type} cluster")

    for index, value in enumerate(times):
        plt.text(value + 0.5, index, f"{value:.2f}s", va='center', fontsize=10)

    plt.gca().invert_yaxis()
    plt.savefig(f"benchmark_results_{cluster_type}.png")


def update_yaml(file_path: str,
                cluster_type: str,
                output_file: str,
                **kwargs) -> str:
    """Update the YAML file with new benchmarking parameters and save it."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    if cluster_type == "slurm":
        cores = kwargs.get("cores", 1)
        processes = kwargs.get("processes", 1)
        memory = kwargs.get("memory", 1)
        scale = kwargs.get("scale", 1)
        config["general"]["client"] = {
            "cluster_type": cluster_type,
            "params": {
                "cores": cores,
                "processes": processes,
                "memory": f"{memory}GB",
                "scale": scale,
                "walltime": "00:15:00",
                "queue": "par-single",
            }
        }
    else:
        workers = kwargs.get("workers", 1)
        threads = kwargs.get("threads", 1)
        memory = kwargs.get("memory", 1)
        config["general"]["client"] = {
            "cluster_type": cluster_type,
            "params": {
                "n_workers": workers,
                "threads_per_worker": threads,
                "memory_limit": f"{memory}GB"
            }
        }

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_file

def benchmark_local(benchmark_params: dict,
                    configuration_file: str,
                    logger: logging.Logger) -> list:
    """Handle the benchmark test for LocalCluster.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.
        benchmark_results (list): The benchmark results.

    Returns:
        list: The benchmark results.
    """
    benchmark_results = []
    cluster_type = "local"
    n_workers = benchmark_params.get("n_workers", [1])
    threads_per_worker = benchmark_params.get("threads_per_worker", [1])
    memory_limit = benchmark_params.get("memory_limit", [1])
    for workers, threads, memory in itertools.product(n_workers, threads_per_worker, memory_limit):
        output_file = f"config_{cluster_type}_{workers}_{threads}_{memory}.yaml"

        updated_config_file = update_yaml(
            file_path=configuration_file,
            cluster_type=cluster_type,
            output_file=output_file,
            workers=workers,
            threads=threads,
            memory=memory
        )
        logger.info("Running benchmark test with %s workers, %s threads, %sGB memory", workers, threads, memory)
        start_time = time.perf_counter()
        benchmark_task(updated_config_file, logger)
        end_time = time.perf_counter()
        logger.info("Benchmark test completed")

        benchmark_results.append({
            "workers": workers,
            "threads": threads,
            "memory": memory,
            "time_taken": round(end_time - start_time, 2),
        })
        Path(output_file).unlink()
        gc.collect()
    return benchmark_results

def benchmark_slurm(benchmark_params: dict,
                    configuration_file: str,
                    logger: logging.Logger) -> list:
    """Handle the benchmark test for SLURM.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.

    Returns:
        list: The benchmark results.
    """
    benchmark_results = []
    cluster_type = "slurm"
    cores = benchmark_params.get("cores", [1])
    processes = benchmark_params.get("processes", [1])
    memory = benchmark_params.get("memory", [1])
    scale = benchmark_params.get("scale", [1])
    for core, proc, mem, sc in itertools.product(cores, processes, memory, scale):
        output_file = f"config_{cluster_type}_{core}_{proc}_{mem}_{sc}.yaml"

        updated_config_file = update_yaml(
            file_path=configuration_file,
            cluster_type=cluster_type,
            output_file=output_file,
            core=core,
            proc=proc,
            mem=mem,
            sc=sc
        )

        logger.info("Running benchmark test with %s cores, %s processes, %sGB memory, %s scale", core, proc, mem, sc)
        start_time = time.perf_counter()
        benchmark_task(updated_config_file, logger)
        end_time = time.perf_counter()
        logger.info("Benchmark test completed")

        benchmark_results.append({
            "cpus": core,
            "processes": proc,
            "memory": mem,
            "scale": sc,
            "time_taken": round(end_time - start_time, 2),
        })

    return benchmark_results

def benchmark_handler(benchmark_params: dict,
                      configuration_file: str,
                      logger: logging.Logger) -> None:
    """Handle the benchmark test.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.
    """
    logger.info("Starting benchmark test")
    configuration_file = Path(configuration_file)
    cluster_type = benchmark_params.get("cluster_type")
    if cluster_type == "slurm":
        logger.info("Running benchmark test on SLURM cluster")
        benchmark_results = benchmark_slurm(benchmark_params, configuration_file, logger)
    else:
        logger.info("Running benchmark test on LocalCluster")
        benchmark_results = benchmark_local(benchmark_params, configuration_file, logger)

    logger.info("Benchmark test completed")

    print(json.dumps(benchmark_results))

    with open(f"benchmark_results_{cluster_type}.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)
    logger.info("Test results saved to benchmark_results_%s.json", cluster_type)

    plot_results(benchmark_results, cluster_type)
    logger.info("Plotting benchmark results")
