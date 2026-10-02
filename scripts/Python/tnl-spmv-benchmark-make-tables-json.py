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
from LegacyCounterparts import legacy_counterparts
from BenchmarkTable import load_table, print_formats_and_launch_configs


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
        "PosterHeatmapGraphs.py / PosterCoverageGraphs.py)",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--list-matrices",
        nargs="?",
        const="processed-matrices.txt",
        default=None,
        metavar="FILE",
        help="Write the names of all processed matrices, one per line, to a text file "
        "in the output directory (default file name: processed-matrices.txt)",
    )
    return parser


def analyze_df(df, args, formats, launch_configs, accelerator_devices):
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


def write_matrices_list(df, file_name):
    """
    Write the names of all processed matrices (one per line) to a text file
    """
    names = [str(name) for name in df[("Matrix name", "", "", "", "")]]
    with open(file_name, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"List of {len(names)} processed matrices ({len(set(names))} unique names) written to {file_name}")

def main():
    argparser = get_arg_parser()
    args = argparser.parse_args()

    result, formats, launch_configs, accelerator_devices = load_table(args.input)
    if args.verbose:
        print_formats_and_launch_configs(formats, launch_configs, accelerator_devices)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    if args.list_matrices:
        write_matrices_list(result, args.list_matrices)

    analyze_df(result, args, formats, launch_configs, accelerator_devices)

    os.chdir("..")


if __name__ == "__main__":
    main()
