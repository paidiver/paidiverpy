import argparse
import logging
import os
import sys

from paidiverpy.color_layer.color_layer import ColorLayer
from paidiverpy.open_layer.open_layer import OpenLayer
from paidiverpy.pipeline import Pipeline
from utils import initialise_logging

logger = logging.getLogger(__name__)

def process_action(parser):
    args = parser.parse_args()

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)

    pipeline = Pipeline(config_file_path=args.configuration_file)
    config = pipeline.config

    images = OpenLayer(config=config).import_image()

    if config.edge:
        images = ColorLayer(config=config).edge_detection(images=images)

    return images


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

    initialise_logging()

    parser = argparse.ArgumentParser(description="Paidiverpy image preprocessing")
    parser = add_arguments(parser)

    process_action(parser)
    logging.info("✔ paidiverpy terminated successfully ✔")
    sys.exit(0)

if __name__ == "__main__":
    main()
