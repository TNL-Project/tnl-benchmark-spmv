#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from TNL.BenchmarkLogs import *
import MultiindexCreator as mic
import Speedup
import Report
import BestFormats
import LatexLabels

bw_units = "TB/s"

# launch_configs = {}

legacy_counterparts = {
    ("BiEllpack", "1 TPS"): "Legacy BiEllpack",
    ("ChunkedEllpack", "1 TPS"): "Legacy ChunkedEllpack",
    ("SlicedEllpack", "1 TPS"): "Legacy SlicedEllpack",
    ("Ellpack", "1 TPS"): "Legacy Ellpack",
    ("CSR", "1 TPS"): "Legacy CSR Scalar",
    ("CSR", "Warp per segment"): "Legacy CSR Vector",
    ("CSR", "Light CSR"): "Legacy CSR LightWithoutAtomic",
    ("CSR Adaptive", "Default"): "Legacy CSR Adaptive",
}


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Script for parsing log files from tnl-benchmark-spmv."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input files",
        default=["sparse-matrix-benchmark.log"],
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
        "--max-rows",
        type=int,
        default=-1,
        help="Maximum number of rows to process from the input files",
    )
    parser.add_argument(
        "--draw-graphs",
        help="Flag to draw graphs",
        action="store_true",
        default=False
    )
    return parser


def add_to_multiindex(mc, format, device, launch_config, launch_configs):
    if (format in ["CSR", "Hypre", "Ginkgo"]) and device == "CPU":
        # For these formats on CPU we want to compute parallel efficiency
        if launch_config == "1 threads":
            bm_data = ["bandwidth", "time mean"]
        else:
            bm_data = ["bandwidth", "time mean", "speed-up", "eff."]
        if format in ["Hypre", "Ginkgo"]:
            bm_data.append("TNL speed-up")
        for data in bm_data:
            mc.add_entry([format, "CPU", launch_config, data])
            print(f"   >>> {format} CPU {launch_config} {data}")
    else:
        # Here we add all the other formats
        if (format, device) in launch_configs:
            for data in [
                "bandwidth",
                "time mean",
                "diff.max",
            ]:
                mc.add_entry([format, device, launch_config, data])
                print(f"   >>> {format} {device} {launch_config} {data}")
        # If there is a legacy counterpart for the format we add speed-up to compare both
        legacy_format = legacy_counterparts.get(format)
        if legacy_format:
            mc.add_entry([format, device, launch_config, "speed-up", legacy_format])
            print(f"   >>> {format} {device} {launch_config} speed-up {legacy_format}")

        # Here we add speed-up comparisons  with cusparse, CSR on CPU, Hypre and Ginkgo Libraries
        if device != "CPU" and not format in ["cusparse"]:
            for speedup in ["cusparse", "CSR CPU", "Hypre", "Ginkgo"]:
                if speedup != format:
                    mc.add_entry([format, device, launch_config, "speed-up", speedup])
                    print(f"   >>> {format} {device} {launch_config} speed-up {speedup}")
        # Add speedup of CSR Light compared to Light SpMV
        if format == "CSR" and launch_config == "Light CSR":
            mc.add_entry([format, device, "Light CSR", "speed-up", "LightSpMV Vector"])
            print(f"   >>> {format} {device} {launch_config} speed-up LightSpMV Vector")

        # Here we add speed-up comparisons for Binary,Symmetric and Sorted formats
        if device != "CPU":
            if "Binary" in format:
                mc.add_entry([format, device, launch_config, "speed-up", "non-binary"])
                print(f"   >>> {format} {device} {launch_config} speed-up non-binary")
            if "Symmetric" in format:
                mc.add_entry(
                    [format, device, launch_config, "speed-up", "non-symmetric"]
                )
                print(f"   >>> {format} {device} {launch_config} speed-up non-symmetric")
            if "Sorted" in format:
                mc.add_entry([format, device, launch_config, "speed-up", "non-sorted"])
                print(f"   >>> {format} {device} {launch_config} speed-up non-sorted")
            if (format, launch_config) in legacy_counterparts:
                mc.add_entry(
                    [
                        format,
                        device,
                        launch_config,
                        "speed-up",
                        legacy_counterparts[(format, launch_config)],
                    ]
                )
                print(
                    f"   >>> {format} {device} {launch_config} speed-up {legacy_counterparts[(format, launch_config)]}"
                )


def get_multiindex(input_df, formats, launch_configs, accelerator_devices):
    """
    Create index for the table.
    """
    mc = mic.MultiindexCreator(5)
    mc.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])

    for format in formats:
        for device in ["CPU"] + accelerator_devices:
            if (format, device) in launch_configs:
                for launch_config in launch_configs[(format, device)]:
                    print(f"Adding to multiindex: {format} {device} {launch_config}")
                    add_to_multiindex(mc, format, device, launch_config, launch_configs)

            # Finaly we add column with the format exhibiting the best performance for given matrix
            if (
                format == "CSR Light Automatic" or format == "CSR Light Automatic Light"
            ) and device != "CPU":
                mc.add_entry([format, device, "", "speed-up", "LightSpMV Vector"])
            if format == "CSR Light Best" and device != "CPU":
                mc.add_entry(["CSR Light Best", device, "", "TPS"])

    return mc.get_multiindex()


def convert_data_frame(input_df, multicolumns, df_data, begin_idx=0, end_idx=-1):
    """
    Convert input table to a wide-format DataFrame using multiindex columns.
    Uses vectorized melt + pivot instead of row-by-row iteration.
    """
    if end_idx == -1:
        end_idx = len(input_df.index)
    matrix_names = input_df["matrix name"].drop_duplicates().tolist()
    matrix_names = matrix_names[begin_idx:end_idx]

    if not matrix_names:
        return pd.DataFrame(columns=multicolumns)

    df = input_df[input_df["matrix name"].isin(matrix_names)].copy()

    df["bandwidth"] = pd.to_numeric(df["bandwidth"], errors="coerce")
    if bw_units == "TB/s":
        df["bandwidth"] = df["bandwidth"] / 1024
    df["time mean"] = pd.to_numeric(df["time mean"], errors="coerce")
    df["CSR Diff.Max"] = pd.to_numeric(df["CSR Diff.Max"], errors="coerce")

    cpu_no_diff_mask = (df["performer"] == "CPU") & (df["format"].isin(["CSR", "Ginkgo", "Hypre"]))
    df.loc[cpu_no_diff_mask, "CSR Diff.Max"] = float("nan")

    melted = df.melt(
        id_vars=["matrix name", "format", "performer", "launch cfg."],
        value_vars=["bandwidth", "time mean", "CSR Diff.Max"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {"bandwidth": "bandwidth", "time mean": "time mean", "CSR Diff.Max": "diff.max"}
    melted["metric"] = melted["metric"].map(metric_map)
    melted = melted.dropna(subset=["value"])

    result = melted.pivot_table(
        index="matrix name",
        columns=["format", "performer", "launch cfg.", "metric"],
        values="value",
        aggfunc="first",
    )

    if not result.columns.empty:
        result.columns = pd.MultiIndex.from_tuples([(*col, "") for col in result.columns])

    for col in result.columns:
        if len(col) > 3 and col[3] == "diff.max":
            vals = result[col]
            if vals.dtype == np.float64:
                result[col] = vals.astype(object)
                mask = vals.notna() & (vals == vals.fillna(0.0).astype(int))
                result.loc[mask, col] = vals[mask].astype(int).astype(object)

    metadata = df.drop_duplicates("matrix name").set_index("matrix name")
    metadata = metadata.reindex(matrix_names)

    result[("Matrix name", "", "", "", "")] = metadata.index
    result[("rows", "", "", "", "")] = metadata["rows"].values
    result[("columns", "", "", "", "")] = metadata["columns"].values
    result[("nonzeros per row", "", "", "", "")] = (
        pd.to_numeric(metadata["nonzeros"], errors="coerce")
        / pd.to_numeric(metadata["rows"], errors="coerce")
    ).values

    result = result.reindex(columns=multicolumns)
    result = result.reindex(matrix_names)
    result = result.reset_index(drop=True)

    result.replace("", float("nan"), inplace=True)
    result = result.copy()
    return result


def parse_input_files(file_list):
    """
    Parse input files and return a single dataframe
    """
    input_df = pd.DataFrame()
    for file in file_list:
        df = get_benchmark_dataframe(file)
        input_df = pd.concat([input_df, df])
    return input_df


def get_formats(input_df):
    """
    Get list of formats from the input dataframe
    """
    formats = list(
        set(input_df["format"].values.tolist())
    )  # list of all formats in the benchmark results
    if "CSR Light Automatic" in formats:
        formats.remove("CSR Light Automatic")
    if "Binary CSR Light Automatic" in formats:
        formats.remove("Binary CSR Light Automatic")
    if "Symmetric CSR Light Automatic" in formats:
        formats.remove("Symmetric CSR Light Automatic")
    if "Symmetric Binary CSR Light Automatic" in formats:
        formats.remove("Symmetric Binary CSR Light Automatic")
    formats.append("CSR Best")
    formats.sort()
    return formats


def get_accelerator_devices(input_df):
    """
    Get the sorted list of accelerator backends (everything but "CPU") present
    in the "performer" column, e.g. "CUDA", "HIP", "SYCL".
    """
    return sorted(set(input_df["performer"].unique()) - {"CPU"})


def get_launch_configs(input_df, accelerator_devices):
    """
    Get list of launch configurations from the input dataframe
    """
    launch_configs = {}
    in_idx = 0
    while in_idx < len(input_df.index):
        row = input_df.iloc[in_idx]
        format = row["format"]
        device = row["performer"]
        launch_cfg = row["launch cfg."]
        if (format, device) not in launch_configs:
            launch_configs[(format, device)] = []
        if launch_cfg not in launch_configs[(format, device)]:
            launch_configs[(format, device)].append(launch_cfg)
        in_idx += 1
    for device in accelerator_devices:
        launch_configs[("CSR Best", device)] = []
        launch_configs[("CSR Best", device)].append("")
    launch_configs[("CSR Best", "CPU")] = []
    launch_configs[("CSR Best", "CPU")].append("")
    return launch_configs


def print_formats_and_launch_configs(formats, launch_configs, accelerator_devices):
    """
    Print formats and their launch configurations.
    """
    print(f"Formats: {formats}")
    for format in formats:
        for device in ["CPU"] + accelerator_devices:
            if (format, device) in launch_configs:
                print(
                    f"Launch configs for {format} on {device}: {launch_configs[(format, device)]}"
                )


def analyze_df(df, args, formats, launch_configs, accelerator_devices):
    """
    Analyze the dataframe and generate reports.
    """
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

    print(f"Parsing input files: {args.input}")
    input_df = parse_input_files(args.input)
    formats = get_formats(input_df)
    accelerator_devices = get_accelerator_devices(input_df)
    print(f"Accelerator devices found in the data: {accelerator_devices}")
    launch_configs = get_launch_configs(input_df, accelerator_devices)
    if args.verbose:
        print_formats_and_launch_configs(formats, launch_configs, accelerator_devices)

    print("Converting data...")
    multicolumns, df_data = get_multiindex(input_df, formats, launch_configs, accelerator_devices)
    aux_df = pd.DataFrame(df_data, columns=multicolumns, index=[0])
    aux_df.to_html("index.html")
    result = convert_data_frame(
        input_df, multicolumns, df_data, begin_idx=0, end_idx=args.max_rows
    )

    print("Computing speed-ups...")
    speedup_getter = Speedup.Speedup(
        result, formats, launch_configs, legacy_counterparts, accelerator_devices
    )
    result = speedup_getter.compute_speedup()
    result.replace(to_replace=" ", value=np.nan, inplace=True)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    analyze_df(result, args, formats, launch_configs, accelerator_devices)

    for rows_count in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
        print(f"Filtering for rows <= {rows_count}")
        filtered_df = result[result["rows"].astype("int32") <= rows_count]
        if filtered_df.empty:
            print(f"No data for rows <= {rows_count}, skipping analysis.")
            continue
        if not os.path.exists(f"rows-le-{rows_count}"):
            os.mkdir(f"rows-le-{rows_count}")
        os.chdir(f"rows-le-{rows_count}")
        analyze_df(filtered_df, args, formats, launch_configs, accelerator_devices)
        os.chdir("..")

    for rows_count in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
        print(f"Filtering for rows >= {rows_count}")
        filtered_df = result[result["rows"].astype("int32") >= rows_count]
        if filtered_df.empty:
            print(f"No data for rows >= {rows_count}, skipping analysis.")
            continue
        if not os.path.exists(f"rows-ge-{rows_count}"):
            os.mkdir(f"rows-ge-{rows_count}")
        os.chdir(f"rows-ge-{rows_count}")
        analyze_df(filtered_df, args, formats, launch_configs, accelerator_devices)
        os.chdir("..")

    os.chdir("..")


if __name__ == "__main__":
    main()
