"""Main module for the paidiverpy CLI."""

import argparse
import json
import os
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


def load_configuration(configuration_file: str) -> dict[str, object]:
    """Load a YAML configuration file."""
    with Path(configuration_file).open() as config_stream:
        return yaml.safe_load(config_stream) or {}


def get_client_settings(configuration: dict[str, object]) -> tuple[str | None, dict[str, object]]:
    """Extract the client cluster type and parameters from the configuration."""
    general = configuration.get("general") if isinstance(configuration.get("general"), dict) else {}
    client = general.get("client") if isinstance(general, dict) else {}
    if not isinstance(client, dict):
        return None, {}
    return client.get("cluster_type"), client.get("params") or {}


def get_conda_environment(client_params: dict[str, object]) -> str | None:
    """Get the conda/micromamba environment name configured for the batch job."""
    env_name = client_params.get("conda_env") or client_params.get("environment")
    if env_name:
        return str(env_name)
    return os.environ.get("PAYDIVERPY_ENV")


def build_sbatch_directives(client_params: dict[str, object], submit_dir: Path) -> list[str]:
    """Build the Slurm directives for the wrapper script."""
    sbatch_directives: list[str] = [
        "#SBATCH --job-name=paidiverpy-driver",
        f"#SBATCH --chdir={submit_dir}",
        "#SBATCH --export=ALL",
    ]

    queue = client_params.get("queue") or client_params.get("partition")
    if queue:
        sbatch_directives.append(f"#SBATCH --partition={queue}")

    account = client_params.get("account")
    if account:
        sbatch_directives.append(f"#SBATCH --account={account}")

    walltime = client_params.get("walltime")
    if walltime:
        sbatch_directives.append(f"#SBATCH --time={walltime}")

    job_extra_directives = client_params.get("job_extra_directives") or []
    for directive in job_extra_directives:
        text = str(directive).strip()
        if not text:
            continue
        if text.startswith("#SBATCH"):
            sbatch_directives.append(text)
        elif text.startswith("--"):
            sbatch_directives.append(f"#SBATCH {text}")
        else:
            sbatch_directives.append(f"#SBATCH --{text}")

    if not any("--output" in directive for directive in sbatch_directives):
        sbatch_directives.append("#SBATCH --output=paidiverpy-driver-%j.out")
    if not any("--error" in directive for directive in sbatch_directives):
        sbatch_directives.append("#SBATCH --error=paidiverpy-driver-%j.err")

    return sbatch_directives


def build_activation_lines(conda_environment: str, configuration_file: str) -> list[str]:
    """Build the shell lines that activate the requested conda environment."""
    return [
        "if command -v micromamba >/dev/null 2>&1; then",
        "    eval $(micromamba shell hook -s bash)",
        f"    micromamba activate {shlex.quote(conda_environment)}",
        "elif command -v conda >/dev/null 2>&1; then",
        "    source $(conda info --base)/etc/profile.d/conda.sh",
        f"    conda activate {shlex.quote(conda_environment)}",
        "else",
        "    echo 'Neither micromamba nor conda is available in the Slurm job environment.' >&2",
        "    exit 1",
        "fi",
        f"exec paidiverpy -c {configuration_file}",
    ]


def build_sbatch_script(configuration_file: str, configuration: dict[str, object]) -> str:
    """Build the batch wrapper used to launch paidiverpy on Slurm."""
    submit_dir = Path.cwd().resolve()
    config_path = Path(configuration_file).resolve()
    _, client_params = get_client_settings(configuration)
    conda_environment = get_conda_environment(client_params)
    paidiverpy_executable = shutil.which("paidiverpy")
    if not paidiverpy_executable and not conda_environment:
        logger.error("The 'paidiverpy' executable was not found in PATH and no conda environment was configured.")
        sys.exit(1)

    configuration_file = shlex.quote(str(config_path))
    body_lines = (
        build_activation_lines(conda_environment, configuration_file)
        if conda_environment
        else [f"exec {shlex.quote(paidiverpy_executable)} -c {configuration_file}"]
    )

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *build_sbatch_directives(client_params, submit_dir),
            "",
            "set -euo pipefail",
            "export PAYDIVERPY_BATCH_MODE=1",
            *body_lines,
            "",
        ],
    )


def submit_sbatch(configuration_file: str) -> None:
    """Generate a Slurm batch file and submit it to the queue."""
    sbatch_executable = shutil.which("sbatch")
    if not sbatch_executable:
        logger.error("The 'sbatch' executable was not found in PATH.")
        sys.exit(1)

    configuration = load_configuration(configuration_file)
    script_content = build_sbatch_script(configuration_file, configuration)

    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as temp_script:
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
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        if stdout:
            logger.error("sbatch stdout: %s", stdout)
        if stderr:
            logger.error("sbatch stderr: %s", stderr)
        logger.error("Failed to submit paidiverpy batch job to Slurm.")
        sys.exit(exc.returncode)
    finally:
        temp_script_path.unlink(missing_ok=True)

    job_id = result.stdout.strip()
    if job_id:
        logger.info("Submitted paidiverpy batch job to Slurm with job id: %s", job_id)
    else:
        logger.info("Submitted paidiverpy batch job to Slurm.")


def run_gui(gui_args: list[str] | None) -> None:
    """Run the GUI server."""
    logger.info("Running the GUI for paidiverpy...")
    panel_executable = shutil.which("panel")
    if not panel_executable:
        logger.error("The 'panel' executable was not found in the system PATH. Please install Panel using 'pip install panel'.")
        sys.exit(1)

    app_path = files("paidiverpy.frontend").joinpath("app.py")

    # Default params
    panel_args = ["--port", "5006", "--address", "0.0.0.0", "--autoreload"]  # noqa: S104
    if gui_args:
        panel_args = gui_args[0].split()

    subprocess.run(  # noqa: S603
        [panel_executable, "serve", str(app_path), *panel_args],
        check=True,
    )


def run_pipeline(configuration_file: str) -> None:
    """Run the main paidiverpy pipeline."""
    pipeline = Pipeline(
        config_file_path=configuration_file,
        logger=logger,
        track_changes=False,
    )
    pipeline.run(close_client=False, save_images=True)
    if pipeline.client:
        pipeline.client.close()


def process_action(parser: argparse.ArgumentParser) -> None:
    """Process the action based on the arguments provided.

    Args:
        parser (argparse.ArgumentParser): The parser to parse the arguments from.
    """
    args = parser.parse_args()

    if args.gui is not None:
        run_gui(args.gui)
        return

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

    if args.validate:
        Configuration.validate_config(args.configuration_file, local=False)
        return

    configuration = load_configuration(args.configuration_file)
    cluster_type, _ = get_client_settings(configuration)
    batch_mode = os.environ.get("PAYDIVERPY_BATCH_MODE") == "1"
    driver_mode = "batch" if batch_mode else "interactive"
    logger.info("Driver mode: %s", driver_mode)
    logger.info("Configured client cluster_type: %s", cluster_type or "none")

    if cluster_type == "slurm" and not batch_mode:
        logger.info("Submitting driver job to Slurm queue.")
        submit_sbatch(args.configuration_file)
        return
    if cluster_type == "slurm" and batch_mode:
        logger.info("Running inside Slurm batch job; Dask SLURM client will be created from config.")

    run_pipeline(args.configuration_file)


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

    return parser


def main() -> None:
    """Main function for the paidiverpy CLI."""
    parser = argparse.ArgumentParser(description="Paidiverpy image preprocessing")
    parser = add_arguments(parser)

    process_action(parser)
    logger.info("✔ paidiverpy terminated successfully ✔")
