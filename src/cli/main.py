"""Main module for the paidiverpy CLI."""

import argparse
import sys
from paidiverpy.pipeline import Pipeline
from paidiverpy.utils import initialise_logging

logger = initialise_logging()


def process_action(parser: argparse.ArgumentParser) -> None:
    """Process the action based on the arguments provided.

    Args:
        parser (argparse.ArgumentParser): The parser to parse the arguments from.
    """
    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)

    pipeline = Pipeline(
        config_file_path=args.configuration_file, logger=logger, track_changes=False,
    )
    pipeline.run()
    pipeline.save_images()


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
        default="./config/config.yaml",
        help="Path to the configuration file 'config.yaml'",
    )
    return parser


def main() -> None:
    """Main function for the paidiverpy CLI."""
    parser = argparse.ArgumentParser(description="Paidiverpy image preprocessing")
    parser = add_arguments(parser)

    process_action(parser)
    logger.info("✔ paidiverpy terminated successfully ✔")
