import argparse
import logging
import sys

from paidiverpy.pipeline import Pipeline
from utils import initialise_logging

logger = initialise_logging()


def process_action(parser):
    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)

    pipeline = Pipeline(config_file_path=args.configuration_file, logger=logger)
    pipeline.run()
    pipeline.save_images()


def add_arguments(parser):
    parser.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="./config/config.yaml",
        help="Path to the configuration file 'config.yaml'",
    )

    return parser


def main():

    parser = argparse.ArgumentParser(description="Paidiverpy image preprocessing")
    parser = add_arguments(parser)

    process_action(parser)
    logger.info("✔ paidiverpy terminated successfully ✔")
