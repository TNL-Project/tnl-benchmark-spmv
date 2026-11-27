#!/usr/bin/python3

"""
Load the complete benchmark table, written by tnl-spmv-benchmark-parse-logs.py,
and draw the charts of the poster at SC2026 from it.
"""

import os
import sys
import argparse

# the modules shared with the other analysis scripts are in the parent directory
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import PosterOverviewGraphs  # noqa: E402
import PosterHeatmapGraphs  # noqa: E402
import PosterCoverageGraphs  # noqa: E402
import CpuLibrariesGraphs  # noqa: E402
import GpuLibrariesGraphs  # noqa: E402
from BenchmarkTable import load_table, print_formats_and_launch_configs  # noqa: E402


def cpu_thread_counts(value):
    """
    Parse the value of --with-cpu-threads: None for "all", otherwise the
    sorted list of the numbers of threads
    """
    if value.strip().lower() == "all":
        return None
    try:
        counts = sorted({int(item) for item in value.split(",")})
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not 'all' or a comma-separated list of integers")
    if not counts or counts[0] < 1:
        raise argparse.ArgumentTypeError("the numbers of threads must be positive")
    return counts


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Script for drawing the charts of the poster at SC2026 from the table written "
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
        "--transposed",
        help="Analyze the results for the transposed matrices instead of the original ones",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v", "--verbose", help="Set verbose output.", action="store_true", default=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for generated files",
        default="spmv-benchmark-poster",
    )
    parser.add_argument(
        "--with-cpu-threads",
        type=cpu_thread_counts,
        default=None,
        metavar="THREADS",
        help="Numbers of threads compared in the graphs of the libraries on the CPU "
        "(CpuLibrariesGraphs.py): 'all' for all benchmarked ones or a comma-separated "
        "list, e.g. 1,4,16,32 (default: all)",
    )
    return parser


def draw_poster_graphs(df, args, formats, launch_configs, accelerator_devices, transposed_table=None):
    """
    Draw the charts of the poster.
    """
    # the comparisons with the other libraries are drawn for the transposed matrices too
    tables = [(df, formats, launch_configs, accelerator_devices, args.transposed)]
    if transposed_table is not None:
        tables.append((*transposed_table, True))
    for table_df, table_formats, table_launch_configs, table_devices, transposed in tables:
        PosterOverviewGraphs.speedup_overview_vs_libraries(
            table_df, table_formats, table_launch_configs, table_devices, transposed=transposed
        )
        PosterOverviewGraphs.speedup_overview_csr_vs_libraries(
            table_df, table_formats, table_launch_configs, table_devices, transposed=transposed
        )
        PosterOverviewGraphs.speedup_overview_ellpack_vs_libraries(
            table_df, table_devices, transposed=transposed
        )
        PosterOverviewGraphs.speedup_overview_ellpack_vs_best_csr(
            table_df, table_devices, transposed=transposed
        )
        PosterOverviewGraphs.speedup_overview_sliced_ellpack_vs_vendor(
            table_df, table_devices, transposed=transposed
        )
        PosterOverviewGraphs.speedup_heatmaps_variants_vs_libraries(
            table_df, table_formats, table_launch_configs, table_devices, transposed=transposed
        )
    PosterOverviewGraphs.speedup_overview_variants(
        df, formats, launch_configs, accelerator_devices
    )
    PosterOverviewGraphs.speedup_overview_csr_binary_vs_nonbinary(
        df, formats, launch_configs, accelerator_devices
    )
    PosterOverviewGraphs.speedup_overview_sorted_segments(df, accelerator_devices)
    PosterHeatmapGraphs.speedup_heatmap_vs_best_csr(df, accelerator_devices)
    PosterHeatmapGraphs.speedup_heatmap_vs_best_csr(
        df, accelerator_devices, include_cpu_csr=True
    )
    PosterHeatmapGraphs.write_speedup_matrices(df, accelerator_devices)
    PosterHeatmapGraphs.write_speedup_matrices(
        df, accelerator_devices, include_cpu_csr=True
    )
    PosterCoverageGraphs.cumulative_coverage_vs_best(df, accelerator_devices)
    CpuLibrariesGraphs.cpu_libraries_graphs(
        df, launch_configs, args.transposed, thread_counts=args.with_cpu_threads
    )
    GpuLibrariesGraphs.gpu_libraries_graphs(df, accelerator_devices, args.transposed)
    if transposed_table is not None:
        transposed_df, _, _, transposed_devices = transposed_table
        GpuLibrariesGraphs.gpu_libraries_graphs(transposed_df, transposed_devices, True)


def main():
    argparser = get_arg_parser()
    args = argparser.parse_args()

    result, formats, launch_configs, accelerator_devices = load_table(args.input, args.transposed)
    if args.verbose:
        print_formats_and_launch_configs(formats, launch_configs, accelerator_devices)

    # the comparisons with the other libraries on the GPUs are made for the transposed matrices too
    transposed_table = None
    if not args.transposed:
        try:
            transposed_table = load_table(args.input, True)
        except ValueError as error:
            print(f"{error} The comparisons with the other libraries are made for the original matrices only.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    draw_poster_graphs(result, args, formats, launch_configs, accelerator_devices, transposed_table)


if __name__ == "__main__":
    main()
