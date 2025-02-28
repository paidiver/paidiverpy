"""This module contains functions to run the benchmark test."""

import gc
import itertools
import json
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from paidiverpy.pipeline.pipeline import Pipeline


def benchmark_task(configuration_file: str, logger: logging.Logger) -> None:
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
    start_time = time.perf_counter()
    pipeline.run()
    end_time = time.perf_counter()
    del pipeline
    gc.collect()
    return start_time, end_time


def plot_results(results: list, cluster_type: str, filename: str) -> None:
    """Plot the benchmark results.

    Args:
        results (list): The list of benchmark results.
        cluster_type (str): The cluster type.
        filename (str): The filename to save the plot.
    """
    if cluster_type == "local":
        labels = [f"{r['workers']} Workers, {r['threads']} Threads, {r['memory']}GB, {r['scale']} Scale" for r in results]
        sorted_indices = np.argsort([r["workers"] * 1000 + r["threads"] * 100 + r["memory"] + r["scale"] for r in results])
        y_label = "Configuration (Workers, Threads, Memory, Scale)"
    elif cluster_type == "slurm":
        labels = [f"{r['cpus']} CPUs, {r['memory']}GB, {r['processes']} Processes, {r['scale']} Scale" for r in results]
        sorted_indices = np.argsort([r["cpus"] * 1000 + r["memory"] + r["processes"] + r["scale"] for r in results])
        y_label = "Configuration (CPUs, Memory, Processes, Scale)"
    else:
        labels = [f"{r['threads']} Threads" for r in results]
        sorted_indices = np.argsort([r["threads"] for r in results])
        y_label = "Configuration (Threads)"

    times = [r["time_taken"] for r in results]

    labels = [labels[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, times, color="skyblue")

    plt.xlabel("Processing Time (seconds)")
    plt.ylabel(y_label)
    plt.title(f"Dask Benchmark on {cluster_type} cluster")

    for index, value in enumerate(times):
        plt.text(value + 0.5, index, f"{value:.2f}s", va="center", fontsize=10)

    plt.gca().invert_yaxis()
    plt.savefig(f"{filename}.png")


def update_yaml(file_path: str, cluster_type: str, output_file: str, n_jobs: int, **kwargs: dict) -> str:
    """Update the YAML file with new benchmarking parameters and save it.

    Args:
        file_path (str): The path to the configuration file.
        cluster_type (str): The cluster type.
        output_file (str): The output file path.
        n_jobs (int): The number of jobs.
        **kwargs: The benchmarking parameters. It should be a dictionary with the following:
            For LocalCluster:
            - workers (int): The number of workers.
            - threads (int): The number of threads.
            - memory (int): The memory limit.
            For SLURM:
            - cores (int): The number of cores.
            - processes (int): The number of processes.
            - memory (int): The memory limit.
            - walltime (str): The walltime.
            - queue (str): The queue name.

    Returns:
        str: The output file path.
    """
    with Path(file_path).open() as f:
        config = yaml.safe_load(f)

    if cluster_type == "slurm":
        cores = kwargs.get("cores", 1)
        processes = kwargs.get("processes", 1)
        memory = kwargs.get("memory", 1)
        walltime = kwargs.get("walltime", "00:15:00")
        queue = kwargs.get("queue", "par-single")
        config["general"]["client"] = {
            "cluster_type": cluster_type,
            "params": {"cores": cores, "processes": processes, "memory": f"{memory}GB", "walltime": walltime, "queue": queue},
        }
    elif cluster_type == "local":
        workers = kwargs.get("workers", 1)
        threads = kwargs.get("threads", 1)
        memory = kwargs.get("memory", 1)
        config["general"]["client"] = {
            "cluster_type": cluster_type,
            "params": {"n_workers": workers, "threads_per_worker": threads, "memory_limit": f"{memory}GB"},
        }
    config["general"]["n_jobs"] = n_jobs

    with Path(output_file).open("w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_file


def benchmark_threads(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> list:
    """Handle the benchmark test for LocalCluster.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.

    Returns:
        list: The benchmark results.
    """
    benchmark_results = []
    n_jobs = benchmark_params.get("n_jobs", [1])
    for n_job in n_jobs:
        output_file = f"config_threads_{n_job}.yml"

        updated_config_file = update_yaml(
            file_path=configuration_file,
            cluster_type=None,
            output_file=output_file,
            n_jobs=n_job,
        )
        logger.info("Running benchmark test with %s threads", n_job)
        start_time, end_time = benchmark_task(updated_config_file, logger)
        logger.info("Benchmark test completed")
        time_taken = round(end_time - start_time, 2)
        logger.info("Time taken: %s seconds", time_taken)
        results = {
            "threads": n_job,
            "time_taken": time_taken,
        }
        benchmark_results.append(results)
        Path(output_file).unlink()
        gc.collect()
    return benchmark_results


def benchmark_local(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> list:
    """Handle the benchmark test for LocalCluster.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.

    Returns:
        list: The benchmark results.
    """
    benchmark_results = []
    cluster_type = "local"
    n_workers = benchmark_params.get("n_workers", [1])
    threads_per_worker = benchmark_params.get("threads_per_worker", [1])
    memory_limit = benchmark_params.get("memory_limit", [1])
    n_jobs = benchmark_params.get("n_jobs", [2])
    for workers, threads, memory, n_job in itertools.product(n_workers, threads_per_worker, memory_limit, n_jobs):
        output_file = f"config_{cluster_type}_{workers}_{threads}_{memory}_{n_jobs}.yml"

        updated_config_file = update_yaml(
            file_path=configuration_file,
            cluster_type=cluster_type,
            output_file=output_file,
            n_jobs=n_job,
            workers=workers,
            threads=threads,
            memory=memory,
        )
        logger.info("Running benchmark test with %s workers, %s threads, %sGB memory, %s scale", workers, threads, memory, n_job)
        start_time, end_time = benchmark_task(updated_config_file, logger)
        logger.info("Benchmark test completed")
        time_taken = round(end_time - start_time, 2)
        logger.info("Time taken: %s seconds", time_taken)
        results = {
            "workers": workers,
            "threads": threads,
            "memory": memory,
            "scale": n_job,
            "time_taken": time_taken,
        }
        benchmark_results.append(results)
        Path(output_file).unlink()
        gc.collect()
    return benchmark_results


def benchmark_slurm(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> list:
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
    walltime = benchmark_params.get("walltime", "00:30:00")
    queue = benchmark_params.get("queue", "par-single")
    n_jobs = benchmark_params.get("n_jobs", [2])
    for core, proc, mem, n_job in itertools.product(cores, processes, memory, n_jobs):
        output_file = f"config_{cluster_type}_{core}_{proc}_{mem}_{n_job}.yml"

        updated_config_file = update_yaml(
            file_path=configuration_file,
            cluster_type=cluster_type,
            output_file=output_file,
            n_jobs=n_job,
            cores=core,
            processes=proc,
            memory=mem,
            walltime=walltime,
            queue=queue,
        )

        logger.info("Running benchmark test with %s cores, %s processes, %sGB memory, %s scale", core, proc, mem, n_job)
        start_time, end_time = benchmark_task(updated_config_file, logger)
        logger.info("Benchmark test completed")
        time_taken = round(end_time - start_time, 2)
        logger.info("Time taken: %s seconds", time_taken)
        results = {
            "cpus": core,
            "processes": proc,
            "memory": mem,
            "scale": n_job,
            "time_taken": time_taken,
        }

        benchmark_results.append(results)

    return benchmark_results


def benchmark_handler(benchmark_params: dict, configuration_file: str, logger: logging.Logger) -> None:
    """Handle the benchmark test.

    Args:
        benchmark_params (dict): The benchmark parameters.
        configuration_file (str): The path to the configuration files.
        logger (logging.Logger): The logger to log messages.
    """
    logger.info("Starting benchmark test")
    configuration_file = Path(configuration_file)
    cluster_type = benchmark_params.get("cluster_type", "threads")
    if cluster_type == "slurm":
        logger.info("Running benchmark test on SLURM cluster")
        benchmark_results = benchmark_slurm(benchmark_params, configuration_file, logger)
    elif cluster_type == "local":
        logger.info("Running benchmark test on LocalCluster")
        benchmark_results = benchmark_local(benchmark_params, configuration_file, logger)
    else:
        logger.info("Running benchmark test using threads")
        benchmark_results = benchmark_threads(benchmark_params, configuration_file, logger)

    logger.info("Benchmark test completed. Test results:")
    for result in benchmark_results:
        logger.info(result)

    filename = f"benchmark_results_{cluster_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    with Path(f"{filename}.json").open("w") as f:
        json.dump(benchmark_results, f, indent=4)
    logger.info("Test results saved to benchmark_results_%s.json", cluster_type)

    plot_results(benchmark_results, cluster_type, filename)
    logger.info("Plotting benchmark results")
