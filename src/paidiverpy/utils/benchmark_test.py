import time
import argparse
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

def benchmark_task():
    # Example task: large array computation
    import dask.array as da
    x = da.random.random((10000, 10000), chunks=(1000, 1000))
    return (x ** 2).mean().compute()

def main(cpus, mem):
    cluster = SLURMCluster(
        cores=cpus,
        memory=f"{mem}GB",
        processes=cpus,
        walltime="01:00:00"
    )
    cluster.scale(cpus)  # Scale to the specified number of CPUs
    client = Client(cluster)

    print(f"Starting benchmark with {cpus} CPUs and {mem}GB memory...")
    start = time.perf_counter()
    result = benchmark_task()
    end = time.perf_counter()

    print(f"Benchmark result: {result}")
    print(f"Time taken: {end - start:.2f} seconds")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", type=int, required=True, help="Number of CPUs")
    parser.add_argument("--mem", type=int, required=True, help="Memory in GB")
    args = parser.parse_args()
    main(args.cpus, args.mem)
