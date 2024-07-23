import argparse
import os
import sys

def add_arguments(obj):
    obj.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="./config/config.yaml",
        help="Path to the configuration file 'config.yaml'",
    )
    obj.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="Input path to load data.",
    )
    obj.add_argument(
        "-o",
        "--output_folder",
        default="results",
        type=str,
        help="Output path to write the results.",
    )


def main(args=None):

    parser = argparse.ArgumentParser(description="Pelagic image preprocessing")
    subparsers = parser.add_subparsers()


if __name__ == "__main__":
    main()
