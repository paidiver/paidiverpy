"""Main module for the paidiverpy CLI."""

import argparse
import json
import sys
from paidiverpy.config.configuration import Configuration
from paidiverpy.pipeline import Pipeline
from paidiverpy.utils.benchmark.benchmark_test import benchmark_handler
from paidiverpy.utils.docker import is_running_in_docker
from paidiverpy.utils.logging_functions import initialise_logging

logger = initialise_logging()


def run_gui(host: str, port: int, reload: bool) -> None:
    """Run the Paidiverpy web dashboard."""
    try:
        from paidiverpy.dashboard import run  # noqa: PLC0415 - optional feature

        run(host=host, port=port, reload=reload)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)


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

    if args.gui:
        run_gui(args.host, args.port, args.reload)
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
        "--gui",
        dest="gui",
        action="store_true",
        help="Launch the Paidiverpy web dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind address (used with --gui).")
    parser.add_argument("--port", default=8792, type=int, help="Dashboard port (used with --gui).")
    parser.add_argument("--reload", action="store_true", help="Reload the dashboard when source files change.")

    return parser


def main() -> None:
    """Main function for the paidiverpy CLI."""
    parser = argparse.ArgumentParser(description="Paidiverpy image preprocessing")
    parser = add_arguments(parser)

    process_action(parser)
    logger.info("✔ paidiverpy terminated successfully ✔")
