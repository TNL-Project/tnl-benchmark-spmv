#!/usr/bin/python3

"""
Load the complete benchmark table, written by tnl-spmv-benchmark-parse-logs.py,
and generate the reports and graphs from it.
"""

import os
import argparse
import Report
import BestFormats
import PosterOverviewGraphs
import PosterHeatmapGraphs
import PosterCoverageGraphs
import CpuLibrariesGraphs
import GpuLibrariesGraphs
from LegacyCounterparts import legacy_counterparts
from BenchmarkTable import load_table, print_formats_and_launch_configs


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
        description="Script for generating reports and graphs from the table written "
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
        default="spmv-benchmark-report",
    )
    parser.add_argument(
        "--draw-graphs",
        help="Flag to draw graphs",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--poster-graphs",
        help="Flag to draw ad hoc poster graphs (see PosterOverviewGraphs.py / "
        "PosterHeatmapGraphs.py / PosterCoverageGraphs.py / CpuLibrariesGraphs.py / GpuLibrariesGraphs.py)",
        action="store_true",
        default=False
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


def analyze_df(df, args, formats, launch_configs, accelerator_devices, transposed_table=None):
    """
    Analyze the dataframe and generate reports.
    """
    if args.poster_graphs:
        PosterOverviewGraphs.speedup_overview_vs_libraries(
            df, formats, launch_configs, accelerator_devices
        )
        PosterOverviewGraphs.speedup_overview_csr_vs_libraries(
            df, formats, launch_configs, accelerator_devices
        )
        PosterOverviewGraphs.speedup_overview_sliced_ellpack_vs_vendor(df, accelerator_devices)
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

    print("Writting to file sparse-matrix-benchmark-test-processed.html ... ")
    df.sort_index(inplace=True)
    df.to_html("sparse-matrix-benchmark-test-processed.html")

    print("Writting to HTML file...")
    df.sort_index(inplace=True)
    df.to_html(f"output.html")

    if args.draw_graphs:
        report = Report.Report(
            df,
            formats,
            launch_configs,
            legacy_counterparts,
            accelerator_devices,
        )
        report.write()

    bestFormats = BestFormats.BestFormats(df, formats, launch_configs, accelerator_devices)
    bestFormats.write()
    bestFormats.count_best_formats("best-formats-report_txt")


def main():
    argparser = get_arg_parser()
    args = argparser.parse_args()

    result, formats, launch_configs, accelerator_devices = load_table(args.input, args.transposed)
    if args.verbose:
        print_formats_and_launch_configs(formats, launch_configs, accelerator_devices)

    # the GPU libraries graphs are drawn for the transposed matrices too
    transposed_table = None
    if args.poster_graphs and not args.transposed:
        try:
            transposed_table = load_table(args.input, True)
        except ValueError as error:
            print(f"{error} The GPU libraries graphs are drawn for the original matrices only.")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    analyze_df(result, args, formats, launch_configs, accelerator_devices, transposed_table)

    os.chdir("..")


if __name__ == "__main__":
    main()
