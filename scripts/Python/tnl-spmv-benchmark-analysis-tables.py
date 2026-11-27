#!/usr/bin/python3

"""
Load the complete benchmark table, written by tnl-spmv-benchmark-parse-logs.py,
and write the tables analyzing the results (see AnalysisTables.py) for the
original and the transposed matrices.
"""

import os
import argparse

import AnalysisTables
from BenchmarkTable import load_table, print_formats_and_launch_configs


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Script for writing the tables analyzing the results from the table written "
        "by tnl-spmv-benchmark-parse-logs.py."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="spmv-benchmark-table.pkl",
        metavar="FILE",
        help="Binary (pickle) file with the complete table written by "
        "tnl-spmv-benchmark-parse-logs.py (default: spmv-benchmark-table.pkl)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for the tables, they are written to its subdirectory "
        f"{AnalysisTables.TABLES_DIR} (default: spmv-benchmark-report)",
        default="spmv-benchmark-report",
    )
    parser.add_argument(
        "-v", "--verbose", help="Set verbose output.", action="store_true", default=True
    )
    return parser


def main():
    args = get_arg_parser().parse_args()

    tables = []
    for transposed in [False, True]:
        try:
            df, formats, launch_configs, accelerator_devices = load_table(args.input, transposed)
        except ValueError as error:
            print(error)
            continue
        if args.verbose:
            print_formats_and_launch_configs(formats, launch_configs, accelerator_devices)
        tables.append((df, accelerator_devices, transposed))

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    for df, accelerator_devices, transposed in tables:
        AnalysisTables.write_tables(df, accelerator_devices, transposed)


if __name__ == "__main__":
    main()
