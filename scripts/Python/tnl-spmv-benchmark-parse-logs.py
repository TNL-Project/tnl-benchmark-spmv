#!/usr/bin/python3

"""
Parse the log files from tnl-benchmark-spmv, convert them to one big table
(one row per matrix, one column per format/device/launch config/metric) and
compute all speed-ups. The results for the original and the transposed
matrices are processed separately, each giving one table. Both tables are
stored in a binary (pickle) file, which tnl-spmv-benchmark-make-tables-json.py
then loads to generate the reports and graphs. The tables can be also
exported to HTML or Excel (see --export).
"""

import pandas as pd
import numpy as np
import argparse
from TNL.BenchmarkLogs import *
import MultiindexCreator as mic
import Speedup
import PosterHeatmapGraphs
from VendorLibraries import vendor_format
from LegacyCounterparts import legacy_counterparts
from BenchmarkTable import save_tables, export_tables, print_formats_and_launch_configs

bw_units = "TB/s"


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Script for parsing log files from tnl-benchmark-spmv and storing "
        "the complete table, including all computed speed-ups, in a binary file."
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
        "--output",
        default="spmv-benchmark-table.pkl",
        metavar="FILE",
        help="Binary (pickle) file where the complete table is stored "
        "(default: spmv-benchmark-table.pkl)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Maximum number of rows to process from the input files",
    )
    parser.add_argument(
        "--duplicates",
        choices=["first", "last", "min", "max"],
        default="first",
        help="How to resolve records of the same matrix, format, device and launch config "
        "appearing more than once, e.g. the CPU results which the CUDA and HIP benchmarks "
        "repeat: use the first or the last one in the order of the input files, or the one "
        "with the smaller or the larger median time (default: first)",
    )
    parser.add_argument(
        "--export",
        nargs="+",
        default=[],
        metavar="FILE",
        help="Export the complete tables also to the given files, the format is given by "
        "the extension: .html or .xlsx",
    )
    return parser


def add_to_multiindex(mc, format, device, launch_config, launch_configs):
    """
    Register the output columns for one (format, device, launch_config)
    combination with the MultiindexCreator `mc`. Each call to mc.add_entry()
    below adds one column, addressed later as a 5-tuple
    (format, device, launch_config, metric, reference) - `reference` is only
    meaningful for "speed-up" columns (empty string otherwise) and names what
    the speed-up was computed against, e.g. speed-up vs. the vendor library of
    the device ("cusparse" on CUDA, "hipsparse" on HIP, see
    VendorLibraries.py) or vs. "non-binary" (the same format's non-Binary
    counterpart).

    There are two branches:
      - CSR/Hypre/Ginkgo on CPU: these are the reference solution, so they
        have no diff.max.
      - everything else: bandwidth/time/diff.max, plus whichever speed-up
        comparisons apply to this format (see the comments below).
    In both branches, CPU runs with "N threads" launch configs get the
    speed-up and the parallel efficiency ("eff.") w.r.t. the "1 thread" run
    of the same format, if there is one.
    """
    cpu_efficiency_data = []
    threads = Speedup.cpu_threads_count(launch_config) if device == "CPU" else None
    if threads is not None and threads > 1 and Speedup.has_single_thread_cpu_run(format, launch_configs):
        cpu_efficiency_data = ["speed-up", "eff."]

    if (format in ["CSR", "Hypre", "Ginkgo"]) and device == "CPU":
        bm_data = ["bandwidth", "time median"] + cpu_efficiency_data
        if format in ["Hypre", "Ginkgo"]:
            bm_data.append("TNL speed-up")
        for data in bm_data:
            mc.add_entry([format, "CPU", launch_config, data])
            #print(f"   >>> {format} CPU {launch_config} {data}")
    else:
        # Here we add all the other formats
        if (format, device) in launch_configs:
            for data in [
                "bandwidth",
                "time median",
                "diff.max",
            ] + cpu_efficiency_data:
                mc.add_entry([format, device, launch_config, data])
                #print(f"   >>> {format} {device} {launch_config} {data}")
        # If there is a legacy counterpart for the format we add speed-up to compare both
        legacy_format = legacy_counterparts.get(format)
        if legacy_format:
            mc.add_entry([format, device, launch_config, "speed-up", legacy_format])
            #print(f"   >>> {format} {device} {launch_config} speed-up {legacy_format}")

        # Here we add speed-up comparisons with the vendor library of the device
        # (cusparse on CUDA, hipsparse on HIP), CSR on CPU, Hypre and Ginkgo Libraries
        vendor = vendor_format(device)
        if device != "CPU" and format != vendor:
            for speedup in [vendor, "CSR CPU", "Hypre", "Ginkgo"]:
                if speedup is not None and speedup != format:
                    mc.add_entry([format, device, launch_config, "speed-up", speedup])
                    #print(f"   >>> {format} {device} {launch_config} speed-up {speedup}")
        # Add speedup of CSR Light compared to Light SpMV
        if format == "CSR" and launch_config == "Light CSR":
            mc.add_entry([format, device, "Light CSR", "speed-up", "LightSpMV Vector"])
            #print(f"   >>> {format} {device} {launch_config} speed-up LightSpMV Vector")

        # Here we add speed-up comparisons for Binary,Symmetric and Sorted formats
        if device != "CPU":
            if "Binary" in format:
                mc.add_entry([format, device, launch_config, "speed-up", "non-binary"])
                #print(f"   >>> {format} {device} {launch_config} speed-up non-binary")
            if "Symmetric" in format:
                mc.add_entry(
                    [format, device, launch_config, "speed-up", "non-symmetric"]
                )
                #print(f"   >>> {format} {device} {launch_config} speed-up non-symmetric")
            if "Sorted" in format:
                mc.add_entry([format, device, launch_config, "speed-up", "non-sorted"])
                #print(f"   >>> {format} {device} {launch_config} speed-up non-sorted")
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
                #print(
                #    f"   >>> {format} {device} {launch_config} speed-up {legacy_counterparts[(format, launch_config)]}"
                #)


def get_multiindex(input_df, formats, launch_configs, accelerator_devices):
    """
    Build the 5-level column MultiIndex for the final wide-format DataFrame:
    one column per (format, device, launch_config, metric, reference) - see
    add_to_multiindex() for what that 5-tuple means.

    Columns are added in three groups, in this order:
      1. per-matrix identity/metadata (name, size, and the statistics from
         MatrixStatistics.h) - level 1 only, the other 4 levels stay "".
      2. one benchmark-result column per (format, device, launch_config)
         actually present in `launch_configs`, via add_to_multiindex().
      3. two special-case columns appended per (format, device): which
         format is fastest for "CSR Light Automatic" ("speed-up" vs.
         LightSpMV), and "CSR Light Best"'s own timing.
    """
    mc = mic.MultiindexCreator(5)
    mc.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros"], ["nonzeros per row"]])
    mc.add_entries([[stat] for stat in PosterHeatmapGraphs.MATRIX_STAT_COLUMNS])

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
    Convert `input_df` - one row per (matrix, format, device, launch config),
    "long" format - into the "wide" table the rest of the script works with:
    one row per matrix, one column per (format, device, launch_config,
    metric, reference) matching `multicolumns` (see get_multiindex()).

    Reshaping is done with melt() + pivot_table() (vectorized) rather than
    iterating rows in Python, which matters here since `input_df` can have
    hundreds of thousands of rows (many matrices x many format/device/
    launch-config combinations x few metrics each).

    begin_idx/end_idx select a slice of matrices (by first-seen order) to
    process, e.g. for --max-rows.
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
    df["time median"] = pd.to_numeric(df["time median"], errors="coerce")
    df["CSR Diff.Max"] = pd.to_numeric(df["CSR Diff.Max"], errors="coerce")

    # CSR/Ginkgo/Hypre on the CPU are themselves the reference solution that
    # every other run's "diff.max" is measured against, so a diff.max value
    # for one of them would just be noise (or a solver artifact) - blank it
    # out rather than let it show up as a spurious accuracy warning.
    cpu_no_diff_mask = (df["performer"] == "CPU") & (df["format"].isin(["CSR", "Ginkgo", "Hypre"]))
    df.loc[cpu_no_diff_mask, "CSR Diff.Max"] = float("nan")

    # Reshape long -> even longer: one row per (matrix, format, performer,
    # launch cfg., metric) with a single "value" column, dropping rows whose
    # value is NaN (e.g. diff.max blanked out just above, or a metric that
    # simply wasn't recorded for that run).
    melted = df.melt(
        id_vars=["matrix name", "format", "performer", "launch cfg."],
        value_vars=["bandwidth", "time median", "CSR Diff.Max"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {"bandwidth": "bandwidth", "time median": "time median", "CSR Diff.Max": "diff.max"}
    melted["metric"] = melted["metric"].map(metric_map)
    melted = melted.dropna(subset=["value"])

    # ... then pivot long -> wide: one row per matrix, one column per
    # (format, performer, launch cfg., metric) combination that actually has
    # data. aggfunc="first" is a formality - each (matrix, format, performer,
    # launch cfg., metric) combination should have exactly one value; if the
    # input ever had duplicates, this silently keeps the first one.
    result = melted.pivot_table(
        index="matrix name",
        columns=["format", "performer", "launch cfg.", "metric"],
        values="value",
        aggfunc="first",
    )

    # pivot_table only produces 4 column levels (format, performer,
    # launch cfg., metric); pad on the 5th ("reference", used only by
    # speed-up columns added below/in Speedup.py) as an empty string so the
    # columns match multicolumns' 5-level shape.
    if not result.columns.empty:
        result.columns = pd.MultiIndex.from_tuples([(*col, "") for col in result.columns])

    # Cosmetic: diff.max values that happen to be whole numbers (most
    # commonly 0.0, i.e. "no difference") are shown as plain ints ("0")
    # instead of floats ("0.0") - purely a display choice, made per-column
    # via an object dtype so the exact (non-integral) diff.max values stay
    # as floats within the same column.
    for col in result.columns:
        if len(col) > 3 and col[3] == "diff.max":
            vals = result[col]
            if vals.dtype == np.float64:
                result[col] = vals.astype(object)
                mask = vals.notna() & (vals == vals.fillna(0.0).astype(int))
                result.loc[mask, col] = vals[mask].astype(int).astype(object)

    # Matrix-level fields (name/size/statistics) are constant across every
    # row of `df` for a given matrix, but not every benchmark writes all of
    # them (e.g. the reference benchmark does not emit MatrixStatistics), so
    # take the first non-null value of each field per matrix and reindex to
    # matrix_names to line rows up with `result` (whose index is already
    # "matrix name", in the same order via pivot_table).
    metadata = df.groupby("matrix name", sort=False).first()
    metadata = metadata.reindex(matrix_names)

    result[("Matrix name", "", "", "", "")] = metadata.index
    result[("rows", "", "", "", "")] = metadata["rows"].values
    result[("columns", "", "", "", "")] = metadata["columns"].values
    result[("nonzeros", "", "", "", "")] = pd.to_numeric(
        metadata["nonzeros"], errors="coerce"
    ).values
    result[("nonzeros per row", "", "", "", "")] = (
        pd.to_numeric(metadata["nonzeros"], errors="coerce")
        / pd.to_numeric(metadata["rows"], errors="coerce")
    ).values
    for stat in PosterHeatmapGraphs.MATRIX_STAT_COLUMNS:
        if stat in metadata.columns:
            result[(stat, "", "", "", "")] = pd.to_numeric(
                metadata[stat], errors="coerce"
            ).values

    # `multicolumns` (from get_multiindex()) is the full, format-independent
    # set of columns the rest of the pipeline expects; `result` only has
    # columns for (format, device, launch_config) combinations that actually
    # had data. reindex(columns=...) adds the missing ones back as all-NaN
    # (e.g. a format that wasn't run on every device) rather than leaving
    # them out, so every output DataFrame has the same column shape.
    result = result.reindex(columns=multicolumns)
    result = result.reindex(matrix_names)
    result = result.reset_index(drop=True)

    result.replace("", float("nan"), inplace=True)
    result = result.copy()
    return result


# The current TNL benchmark logger writes "time_median"/"time_stddev", older
# builds (e.g. of tnl-benchmark-spmv-reference) wrote "time median"/"time
# stddev" - normalize to the latter, which is what the rest of the scripts use.
time_column_aliases = {
    "time median": "time_median",
    "time stddev": "time_stddev",
}


def normalize_time_columns(df):
    """
    Fill the "time median"/"time stddev" columns from their new-style aliases,
    so that logs from both old and new TNL builds can be mixed.
    """
    for column, alias in time_column_aliases.items():
        if alias not in df.columns:
            continue
        if column in df.columns:
            df[column] = df[column].fillna(df[alias])
        else:
            df[column] = df[alias]
        df.drop(columns=[alias], inplace=True)
    return df


def normalize_transposed_column(df):
    """
    Convert the "transposed" column to bool. Logs from older builds have no
    "transposed" column, their results are taken as non-transposed.
    """
    if "transposed" not in df.columns:
        df["transposed"] = False
    else:
        df["transposed"] = df["transposed"].astype(str).str.lower() == "true"
    return df


def parse_input_files(file_list):
    """
    Parse input files and return a single dataframe
    """
    input_df = pd.DataFrame()
    for file in file_list:
        df = get_benchmark_dataframe(file)
        df = normalize_time_columns(df)
        df = normalize_transposed_column(df)
        input_df = pd.concat([input_df, df])
    if "time median" not in input_df.columns:
        raise ValueError("No 'time median' (or 'time_median') column found in the input files.")
    # the index must be unique for resolve_duplicates()
    input_df.reset_index(drop=True, inplace=True)
    check_cpu_builds(input_df)
    transposed_count = input_df["transposed"].sum()
    print(
        f"Parsed {len(input_df.index)} records: {len(input_df.index) - transposed_count} for the "
        f"original matrices, {transposed_count} for the transposed matrices"
    )
    return input_df


def check_cpu_builds(input_df):
    """
    Warn if the CPU results come from several builds of the benchmark (the
    "build" metadata column written by newer builds), since the GPU builds
    usually run on a different machine than the host build and so their CPU
    results are not comparable.
    """
    if "build" not in input_df.columns:
        return
    cpu_builds = sorted(input_df.loc[input_df["performer"] == "CPU", "build"].dropna().unique())
    if len(cpu_builds) > 1:
        print(
            f"WARNING: The CPU results come from several builds of the benchmark {cpu_builds}, "
            "they were probably measured on different machines."
        )


# Records with the same values in these columns are results of the same benchmark.
duplicate_key = ["matrix name", "transposed", "format", "performer", "launch cfg."]


def resolve_duplicates(input_df, policy):
    """
    Keep only one record of each benchmark (see duplicate_key) which appears
    more than once in the input files. This happens mainly for the CPU, since
    the CUDA and HIP builds of the benchmark compute the CSR format (and the
    Binary and Symmetric formats) on the CPU too - older builds always, newer
    ones only with --with-cpu-tests. The policy is one of
      - "first"/"last": the first/last record in the order of the input files,
      - "min"/"max": the record with the smaller/larger median time.
    The order of the remaining records is preserved.
    """
    if "precision" in input_df.columns and input_df["precision"].nunique() > 1:
        print(
            f"WARNING: The input files contain results for several precisions "
            f"{sorted(input_df['precision'].dropna().unique())}, they are treated as duplicates."
        )
    duplicated = input_df.duplicated(subset=duplicate_key, keep=False)
    if not duplicated.any():
        return input_df

    groups = input_df[duplicated].groupby(["performer", "format"]).size()
    print(f"Found {duplicated.sum()} records of benchmarks appearing more than once, keeping the {policy} one:")
    for (performer, format), count in groups.items():
        print(f"   {performer} {format}: {count} records")

    if policy in ["first", "last"]:
        return input_df.drop_duplicates(subset=duplicate_key, keep=policy)
    time = pd.to_numeric(input_df["time median"], errors="coerce")
    order = time.sort_values(ascending=(policy == "min"), na_position="last", kind="stable").index
    return input_df.loc[order].drop_duplicates(subset=duplicate_key, keep="first").sort_index()


def get_formats(input_df):
    """
    Get the list of formats to generate output columns for: every format
    name found in the benchmark log, minus the "CSR Light Automatic" family
    (an internal auto-tuning wrapper around the other CSR Light kernels,
    not itself a distinct storage format worth its own row/column), plus
    the synthetic "CSR Best" format (computed later, not present in the raw
    log) representing "whichever CSR launch config was fastest".
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
    Build {(format, device): [launch_config, ...]} - the set of launch
    configs actually present in the log for each (format, device) pair, in
    first-seen order. get_multiindex()/add_to_multiindex() use this to know
    which columns to create; convert_data_frame() doesn't need it (it derives
    columns straight from the data via pivot_table).

    "CSR Best" is a synthetic format (computed later, see BestFormats.py) so
    it never appears in the log itself; give it a single "" (no-op) launch
    config on every device so it still gets a column added for it.
    """
    launch_configs = {}
    combinations = input_df[["format", "performer", "launch cfg."]].drop_duplicates()
    for format, device, launch_cfg in combinations.itertuples(index=False):
        launch_configs.setdefault((format, device), []).append(launch_cfg)
    for device in accelerator_devices:
        launch_configs[("CSR Best", device)] = []
        launch_configs[("CSR Best", device)].append("")
    launch_configs[("CSR Best", "CPU")] = []
    launch_configs[("CSR Best", "CPU")].append("")
    return launch_configs


def build_table(input_df, args):
    """
    Convert the records of the input files to the multiindex table and compute speed-ups
    """
    formats = get_formats(input_df)
    accelerator_devices = get_accelerator_devices(input_df)
    print(f"Accelerator devices found in the data: {accelerator_devices}")
    launch_configs = get_launch_configs(input_df, accelerator_devices)
    if args.verbose:
        print_formats_and_launch_configs(formats, launch_configs, accelerator_devices)

    print("Converting data...")
    multicolumns, df_data = get_multiindex(input_df, formats, launch_configs, accelerator_devices)
    result = convert_data_frame(
        input_df, multicolumns, df_data, begin_idx=0, end_idx=args.max_rows
    )

    print("Computing speed-ups...")
    speedup_getter = Speedup.Speedup(
        result, formats, launch_configs, legacy_counterparts, accelerator_devices
    )
    result = speedup_getter.compute_speedup()
    result.replace(to_replace=" ", value=np.nan, inplace=True)
    return {
        "df": result,
        "formats": formats,
        "launch_configs": launch_configs,
        "accelerator_devices": accelerator_devices,
    }


def build_tables(args):
    """
    Parse the input files and build one table for the original and one for the
    transposed matrices (None if there are no results for them)
    """
    print(f"Parsing input files: {args.input}")
    input_df = parse_input_files(args.input)
    input_df = resolve_duplicates(input_df, args.duplicates)
    tables = {}
    for name, transposed in [("original", False), ("transposed", True)]:
        records = input_df[input_df["transposed"] == transposed]
        if records.empty:
            print(f"No results for the {name} matrices.")
            tables[name] = None
            continue
        print(f"Building the table for the {name} matrices...")
        tables[name] = build_table(records, args)
    return tables


def main():
    argparser = get_arg_parser()
    args = argparser.parse_args()
    tables = build_tables(args)
    save_tables(args.output, tables)
    for file_name in args.export:
        export_tables(file_name, tables)


if __name__ == "__main__":
    main()
