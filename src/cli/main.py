"""Main module for the paidiverpy CLI."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

import yaml

from paidiverpy.config.configuration import Configuration
from paidiverpy.pipeline import Pipeline
from paidiverpy.utils.benchmark.benchmark_test import benchmark_handler
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.logging_functions import initialise_logging

logger = initialise_logging()


def get_cluster_type(configuration_file: str) -> str | None:
    """Read the configured client cluster type from a YAML config file."""
    with Path(configuration_file).open() as config_stream:
        config = yaml.safe_load(config_stream) or {}
    general = config.get("general", {})
    client = general.get("client", {})
    return client.get("cluster_type")


def submit_slurm_driver(configuration_file: str) -> None:
    """Submit the paidiverpy CLI itself to Slurm and return immediately."""
    sbatch_executable = shutil.which("sbatch")
    if not sbatch_executable:
        logger.error("The 'sbatch' executable was not found in PATH.")
        sys.exit(1)

    submit_dir = Path.cwd().resolve()
    config_path = Path(configuration_file).resolve()

    script_content = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "#SBATCH --job-name=paidiverpy-driver",
            "#SBATCH --output=paidiverpy-driver-%j.out",
            "#SBATCH --error=paidiverpy-driver-%j.err",
            f"cd {shlex.quote(str(submit_dir))}",
            f"paidiverpy -c {shlex.quote(str(config_path))}",
            "",
        ],
    )

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as temp_script:
        temp_script.write(script_content)
        temp_script_path = Path(temp_script.name)

    temp_script_path.chmod(0o700)
    try:
        result = subprocess.run(  # noqa: S603
            [sbatch_executable, "--parsable", str(temp_script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        temp_script_path.unlink(missing_ok=True)

    job_id = result.stdout.strip()
    if job_id:
        logger.info("Submitted paidiverpy driver job to Slurm with job id: %s", job_id)
    else:
        logger.info("Submitted paidiverpy driver job to Slurm.")


def process_action(parser: argparse.ArgumentParser) -> None:
    """Process the action based on the arguments provided.

    Args:
        parser (argparse.ArgumentParser): The parser to parse the arguments from.
    """
    args = parser.parse_args()

    if args.gui is not None:
        logger.info("Running the GUI for paidiverpy...")
        panel_executable = shutil.which("panel")
        if not panel_executable:
            logger.error("The 'panel' executable was not found in the system PATH. Please install Panel using 'pip install panel'.")
            sys.exit(1)
        app_path = files("paidiverpy.frontend").joinpath("app.py")

        # Default params
        gui_args = ["--port", "5006", "--address", "0.0.0.0", "--autoreload"]  # noqa: S104

        if args.gui:
            gui_args = args.gui[0].split()

        subprocess.run(  # noqa: S603
            [panel_executable, "serve", str(app_path), *gui_args],
            check=True,
        )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(2)
    if not args.configuration_file:
        logger.error("Please provide a configuration file.")
        sys.exit(1)
    if args.benchmark_test:
        benchmark_handler(args.benchmark_test, args.configuration_file, logger)
        return

    is_docker = is_running_in_docker()
    if is_docker:
        config_filename = args.configuration_file.split("/")[-1]
        args.configuration_file = f"/app/config_files/{config_filename}"

    if args.submit_only:
        cluster_type = get_cluster_type(args.configuration_file)
        if cluster_type != "slurm":
            logger.error("--submit-only can only be used with a Slurm client configuration.")
            sys.exit(1)
        submit_slurm_driver(args.configuration_file)
        return

    if args.validate:
        Configuration.validate_config(args.configuration_file, local=False)
        return
    pipeline = Pipeline(
        config_file_path=args.configuration_file,
        logger=logger,
        track_changes=False,
    )
    pipeline.run(close_client=False, save_images=True, submit_only=args.submit_only)
    if not args.submit_only and pipeline.client:
        pipeline.client.close()

def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The parser with added arguments.
    """
    parser.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="./config/config.yml",
        help="Path to the configuration file 'config.yml'",
    )

    parser.add_argument(
        "-bt",
        "--benchmark_test",
        dest="benchmark_test",
        type=json.loads,
        help=(
            "OPTIONAL: ONLY FOR BENCHMARK TESTING. Information for benchmark tests "
            "as a JSON string. E.g., "
            '\'{"cluster_type": "slurm", "cores": [1,2,4,8,16,32], "processes": [1,2,4,8,16,32], '
            '"memory": [1,2,4,8,16,32,64], "scale": [1,2,4,8] }\''
        ),
        default={},
    )

    parser.add_argument(
        "-v",
        "--validate",
        dest="validate",
        action="store_true",
        default=False,
        help=("OPTIONAL: ONLY FOR CONFIGURATION FILE CHECKING. Check the configuration file."),
    )

    parser.add_argument(
        "-gui",
        "--gui",
        dest="gui",
        nargs="*",
        help=("OPTIONAL: ONLY FOR RUNNING THE GRAPHICAL USER INTERFACE (GUI) OF PAIDIVERPY."),
    )

    parser.add_argument(
        "-so",
        "--submit-only",
        dest="submit_only",
        action="store_true",
        default=False,
        help=("OPTIONAL: SUBMIT THE PAIDIVERPY DRIVER TO SLURM AND EXIT IMMEDIATELY. Use with Slurm cluster_type only."),
    )

    return parser


def main() -> None:
    """Main function for the paidiverpy CLI."""
    parser = argparse.ArgumentParser(description="Paidiverpy image preprocessing")
    parser = add_arguments(parser)

    process_action(parser)
    logger.info("✔ paidiverpy terminated successfully ✔")
