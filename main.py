import argparse
import os
import sys
from data_preparation.sampling import sampling
from preprocessing.preprocessing import preprocessing


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

    # sampling
    parser_sampling = subparsers.add_parser(
        "sampling",
        help="Prepare dataset ready for training.",
    )
    add_arguments(parser_sampling)
    parser_sampling.set_defaults(func=call_sampling)

    # preprocessing
    parser_preprocessing = subparsers.add_parser(
        "preprocessing",
        help="Apply all preprocessing functions.",
    )
    add_arguments(parser_preprocessing)
    parser_preprocessing.set_defaults(func=call_preprocessing)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)
    args.func(args)


def call_sampling(args):
    sampling(args.configuration_file)


def call_preprocessing(args):
    preprocessing(args.configuration_file, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
