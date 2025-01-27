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
import LatexLabels

bw_units = "TB/s"

formats_devices = []
latex_labels = {}

legacy_couterparts = {
    "BiEllpack BiEllpack": "BiEllpack Legacy",
    "ChunkedElpack ChunkedElpack": "ChunkedEllpack Legacy",
    "SlicedEllpack SlicedEllpack": "SlicedEllpack Legacy",
    "Ellpack Ellpack": "Ellpack Legacy",
    "CSR Scalar": "CSR Legacy Scalar",
    "CSR Vector": "CSR Legacy Vector",
    "CSR Adaptive": "CSR Legacy Adaptive",
    "CSR Light Automatic Light": "CSR Legacy LightWithoutAtomic",
}

"""
The following is a list of formats within which we search for the best performance.
"""
best_formats_list = [
    "Ellpack Ellpack",
    "SlicedEllpack SlicedEllpack",
    "ChunkedEllpack ChunkedEllpack",
    "BiEllpack BiEllpack",
    "CSR Scalar",
    "CSR Vector",
    #    "CSR Light 1",
    #    "CSR Light 2",
    #    "CSR Light 4",
    #    "CSR Light 8",
    #    "CSR Light 16",
    #    "CSR Light 32",
    #    "CSR Light 64",
    #    "CSR Light 128",
    "CSR Light Automatic Light",
    "CSR Adaptive",
    "cusparse",
    "CSR CPU",
    # "Ginkgo CSR",
]


def gaussian(x, a, b, c, d=0):
    return a * math.exp(-((x - b) ** 2) / (2 * c**2)) + d


def color_map(x, width=100, map=[], spread=1):
    width = float(width)
    r = sum(
        [gaussian(x, p[1][0], p[0] * width, width / (spread * len(map))) for p in map]
    )
    g = sum(
        [gaussian(x, p[1][1], p[0] * width, width / (spread * len(map))) for p in map]
    )
    b = sum(
        [gaussian(x, p[1][2], p[0] * width, width / (spread * len(map))) for p in map]
    )
    return min(1.0, r), min(1.0, g), min(1.0, b)


# for x in range(im.size[0]):
#    r, g, b = pixel(x, width=im.size[0], map=heatmap)
#    r, g, b = [int(256*v) for v in (r, g, b)]
#    for y in range(im.size[1]):
#        ld[x, y] = r, g, b


####
# Helper function
def slugify(s):
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def get_multiindex(input_df, formats, threads_num_list):
    """
    Create index for the table.
    """
    mc = mic.MultiindexCreator(4)
    mc.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])

    for format in formats:
        for device in ["CPU", "GPU"]:
            if (format in ["CSR", "Hypre", "Ginkgo"]) and device == "CPU":
                for threads in threads_num_list:
                    if threads == 1:
                        bm_data = [
                            "bandwidth",
                            "time",
                        ]
                    else:
                        bm_data = ["bandwidth", "time", "speed-up", "eff."]
                    if format in ["Hypre", "Ginkgo"]:
                        bm_data.append("TNL speed-up")
                    for (
                        data
                    ) in (
                        bm_data
                    ):  # ,'time','speed-up','non-zeros','stddev','stddev/time','diff.max','diff.l2']:
                        mc.add_entry([format, "CPU", str(threads) + " threads", data])
            else:
                for data in [
                    "bandwidth",
                    "time",
                    "diff.max",
                ]:  # ,'time','speed-up','non-zeros','stddev','stddev/time','diff.max','diff.l2']:
                    mc.add_entry([format, device, data])
            legacy_format = legacy_couterparts.get(format)
            if legacy_format:
                mc.add_entry([format, device, "speed-up", legacy_format])

        if not format in ["cusparse", "CSR"]:
            for speedup in ["cusparse", "CSR CPU", "Hypre", "Ginkgo"]:
                mc.add_entry([format, "GPU", "speed-up", speedup])
        if "Binary" in format:
            mc.add_entry([format, "GPU", "speed-up", "non-binary"])
        if "Symmetric" in format:
            mc.add_entry([format, "GPU", "speed-up", "non-symmetric"])
        if format == "CSR Light Automatic" or format == "CSR Light Automatic Light":
            mc.add_entry([format, "GPU", "speed-up", "LightSpMV Vector"])
        if format == "TNL Best":
            mc.add_entry([format, "GPU", "format"])
        if format == "CSR Light Best":
            mc.add_entry([format, "GPU", "threads per row"])
    return mc.get_multiindex()


def convert_data_frame(input_df, multicolumns, df_data, begin_idx=0, end_idx=-1):
    """
    Convert input table to a better structured one using multiindex
    """
    frames = []
    in_idx = 0
    out_idx = 0
    # max_out_idx = max_rows
    if end_idx == -1:
        end_idx = len(input_df.index)
    best_count = 0
    while in_idx < len(input_df.index) and out_idx < end_idx:
        matrixName = input_df.iloc[in_idx]["matrix name"]
        df_matrix = input_df.loc[input_df["matrix name"] == matrixName]
        if out_idx >= begin_idx:
            print(f"{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName}")
        else:
            print(f"{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName} - SKIP")
        aux_df = pd.DataFrame(df_data, columns=multicolumns, index=[out_idx])
        best_bw_gpu = 0
        best_bw_cpu = 0
        best_csr_light_bw = 0
        best_csr_light_format = "0"
        for index, row in df_matrix.iterrows():
            aux_df.iloc[0]["Matrix name"] = row["matrix name"]
            aux_df.iloc[0]["rows"] = row["rows"]
            aux_df.iloc[0]["columns"] = row["columns"]
            aux_df.iloc[0]["nonzeros per row"] = float(row["nonzeros"]) / float(
                row["rows"]
            )
            current_format = row["format"]
            # print(f"current_format = {current_format}")
            current_device = row["performer"]
            formats_devices.append((current_format, current_device))
            bw = pd.to_numeric(row["bandwidth"], errors="coerce")
            if bw_units == "TB/s":
                bw = bw / 1024
            time = pd.to_numeric(row["time"], errors="coerce")
            diff_max = pd.to_numeric(row["CSR Diff.Max"], errors="coerce")
            if current_device == "CPU" and (
                current_format in ["CSR", "Ginkgo", "Hypre"]
            ):
                threads = str(int(row["threads"]))
                # aux_df.iloc[0][("CSR", "CPU", "1.0 threads", "bandwidth")] = bw
                aux_df.iloc[0][
                    (
                        current_format,
                        current_device,
                        threads + " threads",
                        "bandwidth",
                    )
                ] = bw
                aux_df.iloc[0][
                    (current_format, current_device, threads + " threads", "time")
                ] = time
                if bw > best_bw_cpu:
                    best_bw_cpu = bw
                    best_cpu_threads = threads
            else:
                aux_df.iloc[0][(current_format, current_device, "bandwidth", "")] = bw
                aux_df.iloc[0][(current_format, current_device, "time", "")] = time
                aux_df.iloc[0][
                    (current_format, current_device, "diff.max", "")
                ] = diff_max
            if (
                current_device == "GPU"
                and not "Binary" in current_format
                and not "Symmetric" in current_format
                and not "Legacy" in current_format
                and not "cusparse" in current_format
                and not "LightSpMV" in current_format
                and not "Hybrid" in current_format
                and current_format != "CSR Light Automatic"
                and current_format in best_formats_list
                and bw > best_bw_gpu
            ):
                best_bw_gpu = bw
                best_format_gpu = current_format
            if (
                current_device == "GPU"
                and (
                    current_format == "CSR Light 1"
                    or current_format == "CSR Light 2"
                    or current_format == "CSR Light 4"
                    or current_format == "CSR Light 8"
                    or current_format == "CSR Light 16"
                    or current_format == "CSR Light 32"
                    or current_format == "CSR Light 64"
                    or current_format == "CSR Light 128"
                )
                and bw > best_csr_light_bw
            ):
                best_csr_light_bw = bw
                best_csr_light_format = current_format
            if current_format == "cusparse":
                cusparse_bw = bw
            else:
                cusparse_bw = 0
            # aux_df.iloc[0][(current_format,current_device,'time')]        = row['time']
            # aux_df.iloc[0][(current_format,current_device,'speed-up')]    = row['speedup']
            # aux_df.iloc[0][(current_format,current_device,'non-zeros')]   = row['non-zeros']
            # aux_df.iloc[0][(current_format,current_device,'stddev')]      = row['stddev']
            # aux_df.iloc[0][(current_format,current_device,'stddev/time')] = row['stddev/time']
            # aux_df.iloc[0][(current_format,current_device,'diff.max')]    = row['CSR Diff.Max']
            # aux_df.iloc[0][(current_format,current_device,'diff.l2')]    = row['CSR Diff.L2']
        aux_df.iloc[0][("TNL Best", "GPU", "bandwidth", "")] = best_bw_gpu
        aux_df.iloc[0][("TNL Best", "CPU", "bandwidth", "")] = best_bw_cpu
        if best_bw_gpu > cusparse_bw and best_bw_gpu > best_bw_cpu:
            aux_df.iloc[0][("TNL Best", "GPU", "format", "")] = best_format_gpu
        elif best_bw_cpu > cusparse_bw:
            aux_df.iloc[0][
                ("TNL Best", "GPU", "format", "")
            ] = f"CSR CPU {best_cpu_threads} threads"
        else:
            aux_df.iloc[0][("TNL Best", "GPU", "format", "")] = "cusparse"
        best_count += 1
        best_threads_per_row = best_csr_light_format.replace("CSR Light ", "")
        # print( f'best_csr_light_format = {best_csr_light_format} best_threads_per_row = {best_threads_per_row} \n')
        aux_df.iloc[0][("CSR Light Best", "GPU", "bandwidth", "")] = best_csr_light_bw
        aux_df.iloc[0][("CSR Light Best", "GPU", "threads per row", "")] = int(
            best_threads_per_row
        )
        if out_idx >= begin_idx:
            frames.append(aux_df)
        out_idx = out_idx + 1
        in_idx = in_idx + len(df_matrix.index)
    result = pd.concat(frames)
    result.replace("", float("nan"), inplace=True)
    return result


####
# Parse legacy input file
def parse_legacy(file_name):
    with open(file_name) as f:
        d = json.load(f)
    input_df = json_normalize(d, record_path=["results"])
    # input_df.to_html( "orig-pandas.html" )
    return d


####
# Parse input file
def parse(file_name):
    logFile = open(file_name, "r")

    # read file by lines
    lines = logFile.readlines()

    # drop comments and blank lines
    lines = [line for line in lines if line.strip() and not line.startswith("#")]

    # drop anything before the first metadata block
    # while len(lines) > 0 and not lines[0].startswith(":"):
    #   lines.pop(0)

    df = pd.DataFrame
    # print( lines )
    while len(lines) > 0:
        line = lines.pop(0)
        print(line)
        df.join(pd.read_json(line))
    df.to_html("orig-pandas.html")


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
    "-v", "--verbose", help="Zobrazit více informací", action="store_true"
)

args = parser.parse_args()

print("Parsing input files....")
# d = parse_legacy('sparse-matrix-benchmark.log')
input_df = pd.DataFrame()
for file in args.input:
    df = get_benchmark_dataframe(file)
    input_df = pd.concat([input_df, df])
input_df.to_html("sparse-matrix-benchmark-test.html")


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
formats.append("TNL Best")
formats.append("CSR Light Best")
for format in formats:
    latex_labels[format] = LatexLabels.latex_label(format)
print(f"Formats: {formats}")
print(f"Latex labels: {latex_labels}")

cpu_threads_numbers = list(set(input_df["threads"].values.tolist()))
cpu_threads_numbers = [
    int(x) for x in cpu_threads_numbers if str(x) != "nan" and x != 0
]
# if 0 in threads_num_list:
#    threads_num_list.remove(0)
print(f"CPU threads: {cpu_threads_numbers}")

accepted_formats = ["CSR"]
multicolumns, df_data = get_multiindex(
    input_df,
    formats,
    cpu_threads_numbers,
)
aux_df = pd.DataFrame(df_data, columns=multicolumns, index=[0])
aux_df.to_html("index.html")

print("Converting data...")
result = convert_data_frame(input_df, multicolumns, df_data, begin_idx=0, end_idx=-1)
result.to_html("sparse-matrix-benchmark-test-processed.html")

Speedup.compute_speedup(
    result, formats, cpu_threads_numbers, formats_devices, legacy_couterparts
)
result.replace(to_replace=" ", value=np.nan, inplace=True)

print("Writting to file sparse-matrix-benchmark-test-processed.html ... ")
result.sort_index(inplace=True)
result.to_html("sparse-matrix-benchmark-test-processed.html")


head_size = 25
if not os.path.exists("general"):
    os.mkdir("general")
os.chdir("general")

print("Writting to HTML file...")
result.sort_index(inplace=True)
result.to_html(f"output.html")

report = Report.Report(
    result,
    formats,
    latex_labels,
    formats_devices,
    cpu_threads_numbers,
    legacy_couterparts,
    head_size,
)
report.write()
os.chdir("..")

# for rows_count in [ 10, 100, 1000, 10000, 100000, 1000000, 10000000 ]:
#   filtered_df = result[ result['rows'].astype('int32') <= rows_count ]
#   if not os.path.exists(f'rows-le-{rows_count}'):
#      os.mkdir( f'rows-le-{rows_count}')
#   os.chdir( f'rows-le-{rows_count}')
#   processDf( filtered_df, formats, head_size )
#   os.chdir( '..' )

# for rows_count in [ 10, 100, 1000, 10000, 100000, 1000000, 10000000 ]:
#   filtered_df = result[ result['rows'].astype('int32') >= rows_count ]
#   if not os.path.exists(f'rows-ge-{rows_count}'):
#      os.mkdir( f'rows-ge-{rows_count}')
#   os.chdir( f'rows-ge-{rows_count}')
#   processDf( filtered_df, formats, head_size )
#   os.chdir( '..' )
