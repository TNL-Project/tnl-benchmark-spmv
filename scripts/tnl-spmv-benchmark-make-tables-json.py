#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from TNL.BenchmarkLogs import *
import MultiindexCreator as mic

formats_devices = []

# Latex fonst set-up

# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "sans-serif",
#   "font.sans-serif": ["Helvetica"]})
#
# for Palatino and other serif fonts use:
# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "serif",
#   "font.serif": ["Palatino"],
# })


####
# A map of rgb points in your distribution
# [distance, (r, g, b)]
# distance is percentage from left edge
# https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python/50784012#50784012
heatmap = [
    [0.0, (0.1, 0.1, 1.0)],
    #  [0.20, (0, 0, .5)],
    #  [0.40, (0, .5, 0)],
    [0.40, (0.1, 1.0, 0.1)],
    #   [0.80, (.75, .75, 0)],
    #   [0.90, (1.0, .75, 0)],
    [1.00, (1.0, 0.1, 0.1)],
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


def latexFormatName(name):
    name = name.replace("<", "")
    name = name.replace(">", "")
    name = name.replace("Light  Automatic ", "")
    # print( f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{name}~~~')
    if name == "CSR":
        return "CSR on CPU"
    if name == "cusparse":
        return "cuSPARSE"
    if "SlicedEllpack" in name:
        return name.replace("SlicedEllpack", "Sliced Ellpack")
    if "ChunkedEllpack" in name:
        return name.replace("ChunkedEllpack", "Chunked Ellpack")
    if "BiEllpack" in name:
        return name.replace("BiEllpack", "Bisection Ellpack")
    if "CSR Scalar" in name:
        return name.replace("CSR Scalar", "Scalar CSR")
    if "CSR Vector" in name:
        return name.replace("CSR Vector", "Vector CSR")
    if "CSR Light" in name:
        return name.replace("CSR Light", "Light CSR")
    if "CSR Adaptive" in name:
        return name.replace("CSR Adaptive", "Adaptive CSR")
    return name


#                                                                        brown
matplotlib_fixed_colors = ["blue", "green", "red", "cyan", "magenta", "#663300"]


def get_multiindex(input_df, formats, threads_num_list):
    """
    Create index for the table.
    """
    mc = mic.MultiindexCreator(4)
    mc.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])

    for format in formats:
        for device in ["CPU", "GPU"]:
            if (format == "CSR" or format == "Hypre") and device == "CPU":
                for threads in threads_num_list:
                    if threads == 1:
                        bm_data = [
                            "bandwidth",
                            "time",
                        ]
                    else:
                        bm_data = ["bandwidth", "time", "speed-up", "eff."]
                    if format == "Hypre":
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
        best_bw = 0
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
            time = pd.to_numeric(row["time"], errors="coerce")
            diff_max = pd.to_numeric(row["CSR Diff.Max"], errors="coerce")
            if current_device == "CPU" and (
                current_format == "CSR" or current_format == "Hypre"
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
                and bw > best_bw
            ):
                best_bw = bw
                best_format = current_format
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
        aux_df.iloc[0][("TNL Best", "GPU", "bandwidth", "")] = best_bw
        if best_bw > cusparse_bw:
            aux_df.iloc[0][("TNL Best", "GPU", "format", "")] = best_format
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


def divide_columns(df, in_colA, in_colB, out_col):
    """
    Compute out_col = in_colA / in_colB
    """
    in_colA_list = df[in_colA]
    in_colB_list = df[in_colB]
    out_col_list = []

    for A, B in zip(in_colA_list, in_colB_list):
        div = 0
        try:
            div = A / B
        except:
            div = float("nan")
        out_col_list.append(div)
    df[out_col] = out_col_list


def compute_csr_speedup(df, formats):
    """
    Compute speed-up of particular formats compared to CSR on CPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("CSR", "CPU", "1 threads", "time"),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "CSR CPU"),
                    )


def compute_cusparse_speedup(df, formats):
    """
    Compute speed-up of particular formats compared to Cusparse on GPU and (possibly) Hypre on GPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("cusparse", "GPU", "time", ""),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "cusparse"),
                    )


def compute_hypre_speedup(df, formats):
    """
    Compute speed-up of particular formats compared to Hypre on GPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR", "Hypre"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("Hypre", "GPU", "time", ""),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "Hypre"),
                    )


def compute_ginkgo_speedup(df, formats):
    """
    Compute speed-up of particular formats compared to Ginkgo on GPU and CSR on CPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR", "Ginkgo"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("Ginkgo", "GPU", "time", ""),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "Ginkgo"),
                    )

    # TODO: Compute speedup of Ginkgo CSR on CPU compared to CSR on CPU


def compute_csr_light_speedup(df, formats):
    if "LightSpMV Vector" in formats:
        for light in ["CSR Light Automatic", "CSR Light Automatic Light"]:
            if light in formats:
                if light in formats:
                    divide_columns(
                        df,
                        (light, "GPU", "bandwidth", ""),
                        ("LightSpMV Vector", "GPU", "bandwidth", ""),
                        (light, "GPU", "speed-up", "LightSpMV Vector"),
                    )


def compute_binary_speedup(df, formats):
    for format in formats:
        if "Binary" in format and (format, "GPU") in formats_devices:
            non_binary_format = format.replace("Binary ", "")
            print(f"Adding speed-up of {format} vs {non_binary_format}")
            divide_columns(
                df,
                (format, "GPU", "time"),
                (non_binary_format, "GPU", "time"),
                (format, "GPU", "speed-up", "non-binary"),
            )


def compute_symmetric_speedup(df, formats):
    for format in formats:
        if "Symmetric" in format:
            if (format, "GPU") in formats_devices:
                non_symmetric_format = format.replace("Symmetric ", "")
                print(f"Adding speed-up of {format} vs {non_symmetric_format}")
                divide_columns(
                    df,
                    (non_symmetric_format, "GPU", "time"),
                    (format, "GPU", "time"),
                    (format, "GPU", "speed-up", "non-symmetric"),
                )


def compute_cpu_speedup(df, formats, threads_num_list):
    for format in ["CSR", "Hypre"]:
        if format in formats:
            sequential_times_list = df[(format, "CPU", "1 threads", "time")]
            speedup_list = []
            efficiency_list = []
            for threads in threads_num_list:
                if threads == 1:
                    continue
                parallel_times_list = df[
                    (format, "CPU", str(threads) + " threads", "time")
                ]
                # print( sequential_times_list)
                # print( parallel_times_list)
                for sequential_time, parallel_time in zip(
                    sequential_times_list, parallel_times_list
                ):
                    # print( sequential_time )
                    # print( parallel_time )
                    speedup = float(sequential_time) / float(parallel_time)
                    efficiency = speedup / threads
                    speedup_list.append(speedup)
                    efficiency_list.append(efficiency)
                df[(format, "CPU", str(threads) + " threads", "speed-up")] = (
                    speedup_list
                )
                df[(format, "CPU", str(threads) + " threads", "eff.")] = efficiency_list
                speedup_list.clear()
                efficiency_list.clear()


def compute_hypre_cpu_speedup(df, formats, threads_num_list):
    for threads in threads_num_list:
        divide_columns(
            df,
            ("Hypre", "CPU", str(threads) + " threads", "time"),
            ("CSR", "CPU", str(threads) + " threads", "time"),
            ("Hypre", "CPU", str(threads) + " threads", "TNL speed-up"),
        )


def compute_speedup(df, formats, threads_num_list):
    compute_csr_speedup(df, formats)
    if "cusparse" in formats:
        compute_cusparse_speedup(df, formats)
    if "Hypre" in formats:
        compute_hypre_speedup(df, formats)
    if "Ginkgo" in formats:
        compute_ginkgo_speedup(df, formats)
    compute_csr_light_speedup(df, formats)
    compute_binary_speedup(df, formats)
    compute_symmetric_speedup(df, formats)
    compute_cpu_speedup(df, formats, threads_num_list)
    if ("Hypre", "CPU") in formats_devices:
        compute_hypre_cpu_speedup(df, formats, threads_num_list)


def draw_profiles(
    formats,
    profiles,
    xlabel,
    ylabel,
    filename,
    legend_loc="upper right",
    bar="none",
    yscale="linear",
):
    """
    Draw several profiles into one figure
    formats - list of formats to be drawn
    profiles - dictionary with profiles data
    xlabel - label of x axis
    ylabel - label of y axis
    filename - name of the output file
    legend_loc - location of the legend
    bar - name of the bar to be drawn. Bar is drawn as a line with value 1.
        It serves for better visualization of speed-up (i.e. where it is larger or smaller than one).
        If bar is set to 'none', no bar is drawn.
    yscale - scale of y axis ('linear' or 'log')
    """
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    latexNames = []
    size = 1
    color_idx = 0
    for format in formats:
        t = np.arange(len(profiles[format]))
        if color_idx < len(matplotlib_fixed_colors):
            axs.plot(
                t,
                profiles[format],
                "-o",
                ms=1,
                lw=1,
                color=matplotlib_fixed_colors[color_idx],
            )
            color_idx = color_idx + 1
        else:
            axs.plot(t, profiles[format], "-o", ms=1, lw=1)
        size = len(profiles[format])
        latexNames.append(latexFormatName(format))
    if bar != "none":
        # print( f'size = {size}' )
        bar_data = np.full(size, 1)
        axs.plot(t, bar_data, "-", ms=1, lw=1.5)
        if bar != "":
            latexNames.append(bar)

    axs.legend(latexNames, loc=legend_loc)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_yscale(yscale)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )
    plt.savefig(filename)
    plt.close(fig)


def effective_bw_profile(df, formats, head_size=10):
    """
    Writes figures and HTML tables of effective bandwidth profiles.
    """
    if not os.path.exists("BW-profile"):
        os.mkdir("BW-profile")
    profiles = {}
    color_idx = 0
    print(formats)
    for format in formats:
        print(f"Writing BW profile of {format}")
        if format == "CSR":
            df.sort_values(
                by=[("CSR", "CPU", "1 threads", "bandwidth")],
                inplace=True,
                ascending=False,
            )
            profiles[format] = df[("CSR", "CPU", "1 threads", "bandwidth")].copy()
        else:
            df.sort_values(
                by=[(format, "GPU", "bandwidth", "")], inplace=True, ascending=False
            )
            profiles[format] = df[(format, "GPU", "bandwidth", "")].copy()
        print(f"Writing BW profile of {format}")
        draw_profiles(
            [format],
            profiles,
            xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} performance",
            ylabel="Effective bandwidth in GB/sec",
            filename=f"BW-profile/{format}.pdf",
            legend_loc="upper right",
            bar="none",
            yscale="linear",
        )
        draw_profiles(
            [format],
            profiles,
            xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} performance",
            ylabel="Effective bandwidth in GB/sec",
            filename=f"BW-profile/{format}-log.pdf",
            legend_loc="upper right",
            bar="none",
            yscale="log",
        )
        copy_df = df.copy()
        for f in formats:
            if not f in ["cusparse", "CSR", format]:
                copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
        copy_df.sort_index(inplace=True)
        copy_df.to_html(f"BW-profile/{format}.html")

    # Draw ellpack formats profiles
    current_formats = []
    xlabel = "Matrix number - sorted by particular formats effective bandwidth"
    ylabel = "Effective bandwidth in GB/sec"
    for format in formats:
        if (
            (
                "Ellpack" in format
                and not "Binary" in format
                and not "Symmetric" in format
                and not "Legacy" in format
            )
            or format == "CSR"
            or format == "cusparse"
        ):
            current_formats.append(format)
    draw_profiles(
        current_formats,
        profiles,
        xlabel,
        ylabel,
        "ellpack-profiles-bw.pdf",
        "lower left",
        "none",
    )

    # Draw CSR formats profiles
    current_formats.clear()
    for format in formats:
        if (
            "CSR" in format
            and not "Binary" in format
            and not "Symmetric" in format
            and not "Legacy" in format
            and not "Hybrid" in format
        ) or format == "cusparse":
            current_formats.append(format)
    draw_profiles(
        current_formats,
        profiles,
        xlabel,
        ylabel,
        "csr-profiles-bw.pdf",
        "lower left",
        "none",
    )


def cusparse_comparison(df, formats, head_size=10):
    """
    Writes figures and HTML tables comparing formats with cuSPARSE by the effective bandwidth.
    """
    if "cusparse" in formats:
        if not os.path.exists("Cusparse-bw"):
            os.mkdir("Cusparse-bw")
        ascend_df = df.copy()
        df.sort_values(
            by=[("cusparse", "GPU", "bandwidth", "")], inplace=True, ascending=False
        )
        ascend_df.sort_values(
            by=[("cusparse", "GPU", "bandwidth", "")], inplace=True, ascending=True
        )
        profiles = {}
        profiles["cusparse"] = df[("cusparse", "GPU", "bandwidth", "")].copy()
        for format in formats:
            if not format in ["cusparse", "CSR"]:
                print(f"Writing comparison of {format} and cuSPARSE")
                filtered_df = df.dropna(subset=[(format, "GPU", "bandwidth", "")])
                filtered_ascend_df = ascend_df.dropna(
                    subset=[(format, "GPU", "bandwidth", "")]
                )
                profiles[format] = filtered_df[(format, "GPU", "bandwidth", "")].copy()
                draw_profiles(
                    [format, "cusparse"],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. cuSPARSE performance",
                    ylabel="Effective bandwidth in GB/sec",
                    filename=f"Cusparse-bw/{format}.pdf",
                    legend_loc="upper right",
                    bar="none",
                    yscale="linear",
                )
                draw_profiles(
                    [format, "cusparse"],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. cuSPARSE performance",
                    ylabel="Effective bandwidth in GB/sec",
                    filename=f"Cusparse-bw/{format}-log.pdf",
                    legend_loc="upper right",
                    bar="none",
                    yscale="log",
                )
                copy_df = df.copy()
                for f in formats:
                    if not f in ["cusparse", "CSR", format]:
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                copy_df.to_html(f"Cusparse-bw/{format}.html")


def csr_comparison(df, formats, head_size=10):
    if not os.path.exists("CSR-bw"):
        os.mkdir("CSR-bw")
    profiles = {}
    profiles["CSR"] = df[("CSR", "CPU", "1 threads", "bandwidth")].copy()
    for device in ["CPU", "GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR"]:
                print(f"Writing comparison of {format} and CSR on CPU")
                try:
                    df.sort_values(
                        by=[(format, device, "bandwidth", "")],
                        inplace=True,
                        ascending=False,
                    )
                except:
                    continue
                profiles[format] = df[(format, device, "bandwidth", "")].copy()
                draw_profiles(
                    [format, "CSR"],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. performance of {latexFormatName(format)}",
                    ylabel="Effective bandwidth in GB/sec",
                    filename=f"CSR-bw/{format}.pdf",
                    legend_loc="upper right",
                    bar="none",
                    yscale="linear",
                )

                draw_profiles(
                    [format, "CSR"],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. performance of {latexFormatName(format)}",
                    ylabel="Effective bandwidth in GB/sec",
                    filename=f"CSR-bw/{format}-log.pdf",
                    legend_loc="upper right",
                    bar="none",
                    yscale="log",
                )

                copy_df = df.copy()
                for f in formats:
                    if not f in ["cusparse", "CSR", format]:
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                copy_df.to_html(f"CSR-bw/{format}-{device}.html")


def legacy_formats_comparison(df, formats, head_size=10):
    """
    Comparison of legacy formats with new ones by the effective bandwidth.
    """
    if not os.path.exists("Legacy-bw"):
        os.mkdir("Legacy-bw")
    profiles = {}
    for ref_format, legacy_format in [
        ("Ellpack Ellpack", "Ellpack Legacy"),
        ("SlicedEllpack SlicedEllpack", "SlicedEllpack Legacy"),
        ("ChunkedEllpack ChunkedEllpack", "ChunkedEllpack Legacy"),
        ("BiEllpack BiEllpack", "BiEllpack Legacy"),
        ("CSR Adaptive", "CSR Legacy Adaptive"),
        ("CSR Scalar", "CSR Legacy Scalar"),
        ("CSR Vector", "CSR Legacy Vector"),
    ]:
        if ref_format in formats and legacy_format in formats:
            print(f"Writing comparison of {ref_format} and {legacy_format}")
            ascend_df = df.copy()
            df.sort_values(
                by=[(ref_format, "GPU", "bandwidth", "")], inplace=True, ascending=False
            )
            ascend_df.sort_values(
                by=[(ref_format, "GPU", "bandwidth", "")], inplace=True, ascending=True
            )
            profiles[ref_format] = df[(ref_format, "GPU", "bandwidth", "")].copy()
            profiles[legacy_format] = df[(legacy_format, "GPU", "bandwidth", "")].copy()

            draw_profiles(
                [ref_format, legacy_format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. performance of {latexFormatName(ref_format)}",
                ylabel="Effective bandwidth in GB/sec",
                filename=f"Legacy-bw/{ref_format}.pdf",
                legend_loc="upper right",
                bar="none",
                yscale="linear",
            )

            copy_df = df.copy()
            for f in formats:
                if not f in ["cusparse", "CSR", legacy_format]:
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            copy_df.to_html(f"Legacy-bw/{legacy_format}.html")


def csr_speedup_comparison(df, formats, head_size=10):
    """
    Comparison of speed-up w.r.t. CSR
    """
    if not os.path.exists("CSR-speed-up"):
        os.mkdir("CSR-speed-up")
    for device in ["GPU"]:
        for format in formats:
            profiles = {}
            if not format in ["cusparse", "CSR"]:
                print(f"Writing comparison of speed-up of {format} compared to CSR")
                df["tmp"] = df[(format, device, "bandwidth", "")]
                filtered_df = df.dropna(subset=[("tmp", "", "", "")])
                try:
                    filtered_df.sort_values(
                        by=[(format, device, "speed-up", "CSR CPU")],
                        inplace=True,
                        ascending=False,
                    )
                except:
                    continue

                profiles[format] = filtered_df[
                    (format, device, "speed-up", "CSR CPU")
                ].copy()
                draw_profiles(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. performance of {latexFormatName(format)}",
                    ylabel="Speedup",
                    filename=f"CSR-speed-up/{format}.pdf",
                    legend_loc="upper right",
                    bar="CSR CPU",
                    yscale="linear",
                )
                draw_profiles(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. performance of {latexFormatName(format)}",
                    ylabel="Speedup",
                    filename=f"CSR-speed-up/{format}-log.pdf",
                    legend_loc="upper right",
                    bar="CSR CPU 1 thread",
                    yscale="log",
                )

                copy_df = df.copy()
                for f in formats:
                    if not f in ["cusparse", "CSR", format]:
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                copy_df.to_html(f"CSR-speed-up/{format}-{device}.html")


####
# Comparison of speed-up w.r.t. Cusparse
def cusparse_and_hypre_speedup_comparison(df, formats, head_size=10):
    for comparison in ["cusparse", "Hypre"]:
        if not os.path.exists(f"{comparison}-speed-up"):
            os.mkdir(f"{comparison}-speed-up")
        profiles = {}
        for format in formats:
            if not format in ["cusparse", "CSR", "Hypre"]:
                print(
                    f"Writing comparison of speed-up of {format} ({latexFormatName(format)}) compared to {comparison}"
                )
                df["tmp"] = df[(format, "GPU", "bandwidth", "")]
                filtered_df = df.dropna(subset=[("tmp", "", "", "")])
                filtered_df.sort_values(
                    by=[(format, "GPU", "speed-up", comparison)],
                    inplace=True,
                    ascending=False,
                )
                profiles[format] = filtered_df[
                    (format, "GPU", "speed-up", comparison)
                ].copy()
                fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                size = len(filtered_df[(format, "GPU", "speed-up", comparison)].index)
                t = np.arange(size)
                bar = np.full(size, 1)
                axs.plot(
                    t,
                    filtered_df[(format, "GPU", "speed-up", comparison)],
                    "-o",
                    ms=1,
                    lw=1,
                )
                axs.plot(t, bar, "-", ms=1, lw=1)
                axs.legend([latexFormatName(format), comparison], loc="upper right")
                axs.set_ylabel("Speedup")
                axs.set_xlabel(
                    f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up"
                )
                plt.rcParams.update(
                    {
                        "text.usetex": True,
                        "font.family": "sans-serif",
                        "font.sans-serif": ["Helvetica"],
                    }
                )
                plt.savefig(f"{comparison}-speed-up/{format}.pdf")
                plt.close(fig)

                fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                axs.set_yscale("log")
                axs.plot(
                    t,
                    filtered_df[(format, "GPU", "speed-up", comparison)],
                    "-o",
                    ms=1,
                    lw=1,
                )
                axs.plot(t, bar, "-", ms=1, lw=1)
                axs.legend([latexFormatName(format), comparison], loc="lower left")
                axs.set_xlabel(
                    f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up"
                )
                axs.set_ylabel("Speedup")
                plt.savefig(f"{comparison}-speed-up/{format}-log.pdf")
                plt.close(fig)
                copy_df = df.copy()
                for f in formats:
                    if not f in ["cusparse", "CSR", format, "Hypre"]:
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                copy_df.to_html(f"{comparison}-speed-up/{format}.html")

        # Draw Ellpack formats profiles
        xlabel = f"Matrix number - sorted by particular formats speedup compared to {comparison}"
        ylabel = "Speedup"
        current_formats = []
        for format in formats:
            if (
                "Ellpack" in format
                and not "Symmetric" in format
                and not "Binary" in format
                and not "Legacy" in format
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"ellpack-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )

        current_formats.clear()
        for format in formats:
            if (
                "Ellpack" in format
                and "Symmetric" in format
                and not "Binary" in format
                and not "Legacy" in format
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"symmetric-ellpack-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )

        current_formats.clear()
        for format in formats:
            if (
                "Ellpack" in format
                and not "Symmetric" in format
                and "Binary" in format
                and not "Legacy" in format
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"binary-ellpack-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )

        current_formats.clear()
        for format in formats:
            if (
                "Ellpack" in format
                and "Symmetric" in format
                and "Binary" in format
                and not "Legacy" in format
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"symmetric-binary-ellpack-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )

        # Draw CSR formats profiles
        current_formats.clear()
        for format in formats:
            if (
                "CSR" in format
                and not "Symmetric" in format
                and not "Binary" in format
                and not "Legacy" in format
                and not "Hybrid" in format
                and format != "CSR"
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"csr-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )
        current_formats.clear()
        for format in formats:
            if (
                "CSR" in format
                and "Symmetric" in format
                and not "Binary" in format
                and not "Legacy" in format
                and not "Hybrid" in format
                and format != "CSR"
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"symmetric-csr-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )
        current_formats.clear()

        for format in formats:
            if (
                "CSR" in format
                and not "Symmetric" in format
                and "Binary" in format
                and not "Legacy" in format
                and not "Hybrid" in format
                and format != "CSR"
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"binary-csr-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )
        current_formats.clear()

        for format in formats:
            if (
                "CSR" in format
                and "Symmetric" in format
                and "Binary" in format
                and not "Legacy" in format
                and not "Hybrid" in format
                and format != "CSR"
            ):
                current_formats.append(format)
        if current_formats:
            draw_profiles(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                f"symmetric-binary-csr-profiles-{comparison}-speedup.pdf",
                "upper right",
                comparison,
            )
        current_formats.clear()


####
# Comparison of binary matrices
def binary_matrices_comparison(df, formats, head_size=10):
    if not os.path.exists("binary-speed-up"):
        os.mkdir("binary-speed-up")
    for format in formats:
        profiles = {}
        if "Binary" in format:
            non_binary_format = format.replace("Binary ", "")
            print(f"Writing comparison of speed-up of {format} vs {non_binary_format}")
            # df['tmp'] = df[(format, 'GPU','speed-up','non-binary')]
            filtered_df = df.dropna(
                subset=[(format, "GPU", "speed-up", "non-binary")]
            )  # ('tmp','','','')])
            # print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-binary')]}" )
            ascend_df = filtered_df.copy()
            filtered_df.sort_values(
                by=[(format, "GPU", "speed-up", "non-binary")],
                inplace=True,
                ascending=False,
            )
            ascend_df.sort_values(
                by=[(format, "GPU", "speed-up", "non-binary")],
                inplace=True,
                ascending=True,
            )
            profiles[format] = filtered_df[
                (format, "GPU", "speed-up", "non-binary")
            ].copy()
            draw_profiles(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
                ylabel="Speedup",
                filename=f"binary-speed-up/{format}.pdf",
                legend_loc="upper right",
                bar=f"{latexFormatName(non_binary_format)}",
                yscale="linear",
            )
            draw_profiles(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
                ylabel="Speedup",
                filename=f"binary-speed-up/{format}-log.pdf",
                legend_loc="upper right",
                bar=f"{latexFormatName(non_binary_format)}",
                yscale="log",
            )

            copy_df = filtered_df.copy()
            for f in formats:
                if not f in ["cusparse", "CSR", format, non_binary_format]:
                    # print( f"Droping {f}..." )
                    # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            # head_df.to_html( f"Binary-speed-up/{format}-head.html" )
            copy_df.to_html(f"binary-speed-up/{format}.html")


def symmetric_matrices_comparison(df, formats, head_size=10):
    """
    Comparison of symmetric matrices
    """
    if not os.path.exists("symmetric-speed-up"):
        os.mkdir("symmetric-speed-up")
    for format in formats:
        profiles = {}
        if "Symmetric" in format:
            non_symmetric_format = format.replace("Symmetric ", "")
            print(
                f"Writing comparison of speed-up of {format} vs {non_symmetric_format}"
            )
            # df['tmp'] = df[(format, 'GPU','speed-up','non-symmetric')]
            filtered_df = df.dropna(
                subset=[(format, "GPU", "speed-up", "non-symmetric")]
            )  # ('tmp','','','')])
            # ascend_df = filtered_df.copy()
            # print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-symmetric')]}" )
            filtered_df.sort_values(
                by=[(format, "GPU", "speed-up", "non-symmetric")],
                inplace=True,
                ascending=False,
            )
            # ascend_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=True)

            cusparse_filtered_df = df.dropna(
                subset=[(format, "GPU", "speed-up", "cusparse")]
            )  # ('tmp','','','')])
            cusparse_filtered_df.sort_values(
                by=[(format, "GPU", "speed-up", "cusparse")],
                inplace=True,
                ascending=False,
            )
            profiles[format] = filtered_df[(format, "GPU", "speed-up", "non-symmetric")]
            draw_profiles(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
                ylabel="Speedup",
                filename=f"symmetric-speed-up/{format}.pdf",
                legend_loc="upper right",
                bar=f"{latexFormatName(non_symmetric_format)}",
                yscale="linear",
            )
            draw_profiles(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
                ylabel="Speedup",
                filename=f"symmetric-speed-up/{format}-log.pdf",
                legend_loc="upper right",
                bar=f"{latexFormatName(non_symmetric_format)}",
                yscale="log",
            )

            profiles[format] = cusparse_filtered_df[
                (format, "GPU", "speed-up", "cusparse")
            ]
            draw_profiles(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
                ylabel="Speedup",
                filename=f"symmetric-speed-up/{format}-cusparse.pdf",
                legend_loc="upper right",
                bar="cuSPARSE",
                yscale="linear",
            )

            draw_profiles(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
                ylabel="Speedup",
                filename=f"symmetric-speed-up/{format}-cusparse-log.pdf",
                legend_loc="upper right",
                bar="cuSPARSE",
                yscale="log",
            )

            copy_df = df.copy()
            for f in formats:
                if not f in ["cusparse", "CSR", format, non_symmetric_format]:
                    # print( f"Droping {f}..." )
                    # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            # head_df.to_html( f"Symmetric-speed-up/{format}-head.html" )
            copy_df.sort_values(
                by=[(format, "GPU", "speed-up", "non-symmetric")],
                inplace=True,
                ascending=False,
            )
            copy_df.to_html(f"symmetric-speed-up/{format}.html")
            # copy_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,descending=True)
            # copy_df.to_html( f"Symmetric-speed-up/{format}-sort.html" )


####
# Comparison of speed-up w.r.t. LightSpMV
def csr_light_speedup_comparison(df, head_size=10):
    format = "CSR Ligh Automatic Light"
    print(f"Writing comparison of speed-up of CSR Light compared to LightSPMV")
    if (format, "GPU") in formats_devices:
        profiles = {}
        profiles[format] = df[(format, "GPU", "speed-up", "LightSpMV Vector")].copy()
        draw_profiles(
            [format],
            profiles,
            xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
            ylabel="Speedup",
            filename="LightSpMV-speed-up.pdf",
            legend_loc="upper right",
            bar="LightSpMV",
            yscale="linear",
        )

        draw_profiles(
            [format],
            profiles,
            xlabel=f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up",
            ylabel="Speedup",
            filename="LightSpMV-speed-up-log.pdf",
            legend_loc="upper right",
            bar="LightSpMV",
            yscale="log",
        )

        # df["tmp"] = df[(format, "GPU", "bandwidth", "")]
        # filtered_df = df.dropna(subset=[("tmp", "", "", "")])
        # ascend_df = filtered_df.copy()
        # filtered_df.sort_values(
        #     by=[(format, "GPU", "speed-up", "LightSpMV Vector")],
        #     inplace=True,
        #     ascending=False,
        # )
        # ascend_df.sort_values(
        #     by=[(format, "GPU", "speed-up", "LightSpMV Vector")],
        #     inplace=True,
        #     ascending=True,
        # )
        # fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        # size = len(filtered_df[(format, "GPU", "speed-up", "LightSpMV Vector")].index)
        # t = np.arange(size)
        # bar = np.full(size, 1)
        # axs.plot(
        #     t,
        #     filtered_df[(format, "GPU", "speed-up", "LightSpMV Vector")],
        #     "-o",
        #     ms=1,
        #     lw=1,
        # )
        # axs.plot(t, bar, "-", ms=1, lw=1)
        # axs.legend([latexFormatName(format), "LightSpMV"], loc="upper right")
        # axs.set_ylabel("Speedup")
        # axs.set_xlabel(
        #     f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up"
        # )
        # plt.rcParams.update(
        #     {
        #         "text.usetex": True,
        #         "font.family": "sans-serif",
        #         "font.sans-serif": ["Helvetica"],
        #     }
        # )
        # # for Palatino and other serif fonts use:
        # # plt.rcParams.update({
        # #   "text.usetex": True,
        # #   "font.family": "serif",
        # #   "font.serif": ["Palatino"],
        # # })
        # plt.savefig(f"LightSpMV-speed-up.pdf")
        # plt.close(fig)

        # fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        # axs.set_yscale("log")
        # axs.plot(
        #     t,
        #     filtered_df[(format, "GPU", "speed-up", "LightSpMV Vector")],
        #     "-o",
        #     ms=1,
        #     lw=1,
        # )
        # axs.plot(t, bar, "-", ms=1, lw=1)
        # axs.legend([latexFormatName(format), "LightSpMV"], loc="lower left")
        # axs.set_xlabel(
        #     f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up"
        # )
        # axs.set_ylabel("Speedup")
        # plt.savefig(f"LightSpMV-speed-up-log.pdf")
        # plt.close(fig)
        # head_df = filtered_df.head( head_size )
        # bottom_df = ascend_df.head( head_size )
        copy_df = df.copy()
        for f in formats:
            if not f in ["cusparse", "CSR", format]:
                # print( f"Droping {f}..." )
                # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
        copy_df.sort_index(inplace=True)
        # head_df.to_html( f"LightSpMV-speed-up-head.html" )
        copy_df.to_html(f"LightSpMV-speed-up-bottom.html")


def csr_hypre_cpu_scalability(df, formats, head_size=10):
    cpu_formats = ["CSR"]
    if ("Hypre", "CPU") in formats_devices:
        cpu_formats.append("Hypre")
    for format in cpu_formats:
        for threads in threads_num_list:
            if threads == 1:
                continue
            for metrics in ["speed-up", "eff."]:
                threads_str = str(threads) + " threads"
                print(f"Writing scalability of {format} on CPU")
                df["tmp"] = df[(format, "CPU", threads_str, metrics)]
                filtered_df = df.dropna(subset=[("tmp", "", "", "")])
                try:
                    filtered_df.sort_values(
                        by=[(format, "CPU", threads_str, metrics)],
                        inplace=True,
                        ascending=False,
                    )
                except:
                    continue
                fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                size = len(filtered_df[(format, "CPU", threads_str, metrics)].index)
                t = np.arange(size)
                bar = np.full(size, 1)
                axs.plot(
                    t,
                    filtered_df[(format, "CPU", threads_str, metrics)],
                    "-o",
                    ms=1,
                    lw=1,
                )
                axs.plot(t, bar, "-", ms=1, lw=1)
                axs.legend([latexFormatName(format)], loc="upper right")
                axs.set_ylabel(metrics)
                axs.set_xlabel(
                    f"Matrix number - sorted w.r.t. {latexFormatName(format)} {metrics}"
                )
                plt.rcParams.update(
                    {
                        "text.usetex": True,
                        "font.family": "sans-serif",
                        "font.sans-serif": ["Helvetica"],
                    }
                )
                plt.savefig(f"{format}-{threads}-threads-{metrics}.pdf")
                plt.close(fig)


def hypre_cpu_tnl_speedup_scalability(df, formats, head_size=10):
    if ("Hypre", "CPU") in formats_devices:
        for threads in threads_num_list:
            threads_str = str(threads) + " threads"
            print(f"Writing scalability of {format} on CPU")
            df["tmp"] = df[("Hypre", "CPU", threads_str, "TNL speed-up")]
            filtered_df = df.dropna(subset=[("tmp", "", "", "")])
            try:
                filtered_df.sort_values(
                    by=[("Hypre", "CPU", threads_str, "TNL speed-up")],
                    inplace=True,
                    ascending=False,
                )
            except:
                continue
            fig, axs = plt.subplots(1, 1, figsize=(6, 4))
            size = len(filtered_df[("Hypre", "CPU", threads_str, "TNL speed-up")].index)
            t = np.arange(size)
            bar = np.full(size, 1)
            axs.plot(
                t,
                filtered_df[("Hypre", "CPU", threads_str, "TNL speed-up")],
                "-o",
                ms=1,
                lw=1,
            )
            axs.plot(t, bar, "-", ms=1, lw=1)
            axs.legend(["Hypre"], loc="upper right")
            axs.set_ylabel("Speedup")
            axs.set_xlabel(f"Matrix number - sorted w.r.t. Hypre TNL speed-up")
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Helvetica"],
                }
            )
            plt.savefig(f"Hypre-{threads}-threads-TNL-speed-up.pdf")
            plt.close(fig)


####
# Analyze mapping of CUDA thredads in Light CSR
#
def analyze_light_csr(df, formats):
    sort_df = df.sort_values(
        by=[
            ("CSR Light Best", "GPU", "threads per row", ""),
            ("nonzeros per row", "", "", ""),
        ],
        inplace=False,
        ascending=True,
    )
    for f in formats:
        if not f in ["CSR Light Best"]:
            sort_df.drop(labels=f, axis="columns", level=0, inplace=True)
    sort_df.sort_index(inplace=True)
    sort_df.to_html(f"LightSpMV-Threads-per-row-best.html")
    size = len(sort_df[("nonzeros per row", "", "", "")].index)
    t = np.arange(size)
    fig, axs = plt.subplots(1, 1)
    axs.set_yticks(np.arange(0, 100, step=10))
    axs.plot(t, sort_df[("nonzeros per row", "", "", "")], "-o", ms=1, lw=1)
    axs.plot(
        t, sort_df[("CSR Light Best", "GPU", "threads per row", "")], "-o", ms=1, lw=1
    )
    axs.legend(["Nonzeros per row", "Threads per row"], loc="upper right")
    axs.set_ylabel("CSR Light analysis")
    axs.set_ylim([0, 200])
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )
    plt.savefig(f"LightSpMV-threads-mapping.pdf")
    plt.close(fig)

    profiles = {}
    sort_df = df.sort_values(
        by=[("nonzeros per row", "", "", "")], inplace=False, ascending=True
    )
    formats_list = [
        "CSR Light 1",
        "CSR Light 2",
        "CSR Light 4",
        "CSR Light 8",
        "CSR Light 16",
        "CSR Light 32",
    ]
    if formats_list in formats:
        for format in formats_list:
            sort_df.drop(labels=format, axis="columns", level=0, inplace=True)
            profiles[format] = df[(format, "GPU", "bandwidth", "")].copy()
        sort_df.sort_index(inplace=True)
        sort_df.to_html(f"LightSpMV-Threads-per-row.html")
        draw_profiles(
            formats_list,
            profiles,
            "non-zeros per row",
            "BW",
            "nonzeros-bw.pdf",
            "lower right",
            "none",
        )


def write_colormap(file, max_bw, size, x_position, y_position, standalone=False):
    if standalone:
        file.write("\\documentclass{standalone}\n")
        file.write("\\usepackage[utf8]{inputenc}\n")
        file.write("\\usepackage{tikz}\n")
        file.write("\\begin{document}\n")
        file.write("\\begin{tikzpicture}\n")
    i = 0
    x = x_position
    while i <= max_bw:
        y = y_position + i / max_bw * size
        r, g, b = color_map(i, max_bw, map=heatmap)
        file.write(f"\\definecolor{{color_hm_{i}}}{{rgb}}{{ {r}, {g}, {b} }}; \n")
        file.write(f"\\filldraw[color_hm_{i}] ({x},{y}) circle (2pt); \n")
        i = i + 5
    i = 0
    while i <= max_bw:
        y = y_position + i / max_bw * size
        file.write(
            f"\\filldraw[black] ({x},{y}) circle (1pt) node[anchor=west] {{{i}}}; \n"
        )
        i = i + 400

    if standalone:
        file.write("\\end{tikzpicture}\n")
        file.write("\\end{document}\n")


def write_performance_circle_latex_base(file_name):
    file = open(f"{file_name}-base.tex", "w")
    file.write("\\documentclass{standalone}\n")
    file.write("\\usepackage[utf8]{inputenc}\n")
    file.write("\\usepackage{tikz}\n")
    file.write("\\begin{document}\n")
    file.write("\\begin{tikzpicture}\n")
    file.write(f"\\input{{{file_name}.tex}}\n")
    file.write("\\end{tikzpicture}\n")
    file.write("\\end{document}\n")


#####
# Draw performance circle in tikz
def write_performance_circle(
    df, formats, circle_formats, file_name, scale=1, with_color_map=False
):
    write_performance_circle_latex_base(file_name)
    file = open(f"{file_name}.tex", "w")
    formats_number = 0
    for format in circle_formats:
        if format in formats:
            formats_number += 1

    format_idx = 0
    pos_x = 5 * scale
    pos_y = 5 * scale
    rad = 5 * scale
    formats_pos_x = {}
    formats_pos_y = {}
    for format in circle_formats:
        if format in formats:
            format_angle = (
                math.pi / 2
                - 2 * math.pi / formats_number * format_idx
                - math.pi / formats_number
            )
            if format_angle < 0:
                format_angle = 2 * math.pi + format_angle
            x = pos_x + rad * math.cos(format_angle)
            y = pos_y + rad * math.sin(format_angle)
            formats_pos_x[format] = x
            formats_pos_y[format] = y
            anchor = ""
            if format_angle <= math.pi * 1 / 4 or format_angle > math.pi * 7 / 4:
                anchor = "west"
            if format_angle <= math.pi * 3 / 4 and format_angle > math.pi * 1 / 4:
                anchor = "south"
            if format_angle <= math.pi * 5 / 4 and format_angle > math.pi * 3 / 4:
                anchor = "east"
            if format_angle <= math.pi * 7 / 4 and format_angle > math.pi * 5 / 4:
                anchor = "north"
            # print( f'{format_angle} : {format} -> {anchor} \n' )
            file.write(
                f"\\filldraw[black] ({x},{y}) circle (2pt) node[anchor={anchor}]{{{latexFormatName(format)}}}; \n"
            )
            div_angle = format_angle + math.pi / formats_number
            div_x = pos_x + rad * math.cos(div_angle)
            div_y = pos_y + rad * math.sin(div_angle)
            file.write(f"\\draw [dashed] ({div_x},{div_y}) -- ({pos_x},{pos_y}); \n")
            format_idx += 1
    formats_count = format_idx
    line_idx = 0
    elim = 0
    while line_idx < len(df.index):
        # matrixName = df.iloc[line_idx]['Matrix name']
        sum_bw = 0
        formats_bw = {}
        max_bw = 0
        for format in circle_formats:
            if format in formats:
                format_bw = df.iloc[line_idx][(format, "GPU", "bandwidth", "")]
                formats_bw[format] = format_bw
                # print( f'{matrixName} {format} -> {format_bw}')
                # if format_bw > max_bw:
                sum_bw = sum_bw + format_bw
                if format_bw > max_bw:
                    max_bw = format_bw
        for format in circle_formats:
            if format in formats:
                formats_bw[format] = formats_bw[format] / sum_bw
        format_pos_x = 0
        format_pos_y = 0
        for format in circle_formats:
            if format in formats:
                format_pos_x = format_pos_x + formats_pos_x[format] * formats_bw[format]
                format_pos_y = format_pos_y + formats_pos_y[format] * formats_bw[format]
        if (
            format_pos_x == format_pos_x and format_pos_y == format_pos_y
        ):  # check for NaN
            r, g, b = color_map(max_bw, 1200, map=heatmap)
            file.write(
                f"\\definecolor{{color_{line_idx}}}{{rgb}}{{ {r}, {g}, {b} }} \n"
            )
            file.write(
                f"\\filldraw[color_{line_idx},opacity=0.75] ({format_pos_x},{format_pos_y}) circle (1pt); \n"
            )
        else:
            elim = elim + 1
        line_idx += 1
    if with_color_map:
        write_colormap(file, 1200, 5, 13 * scale, 1.5 * scale, standalone=False)
    os.system(f"pdflatex {file_name}-base.tex")
    print(f"Eliminated formats: {elim}")


def write_performance_circles(df, formats):

    write_performance_circle(
        df,
        formats,
        [
            "cusparse",
            "Ellpack",
            "SlicedEllpack",
            "ChunkedEllpack",
            "BiEllpack",
            "CSR< Scalar >",
            "CSR< Adaptive >",
            "CSR< Vector >",
            "CSR< Light > Automatic Light",
        ],
        "performance-graph",
    )

    scale = 0.6
    aux_df = df
    aux_df.sort_values(
        by=[("SlicedEllpack", "GPU", "bandwidth")], inplace=True, ascending=True
    )
    write_performance_circle(
        aux_df,
        formats,
        ["Ellpack", "ChunkedEllpack", "SlicedEllpack"],
        "performance-graph-ellpacks-1",
        scale,
        with_color_map=False,
    )
    write_performance_circle(
        aux_df,
        formats,
        [
            "BiEllpack",
            "ChunkedEllpack",
            "SlicedEllpack",
        ],
        "performance-graph-ellpacks-2",
        scale,
        with_color_map=False,
    )
    # write_performance_circle( df, formats, ['CSR< Scalar >', 'CSR< Adaptive >', 'CSR< Vector >', 'CSR< Light > Automatic Light'], 'performance-graph-csr-1' )
    aux_df.sort_values(
        by=[("CSR< Light > Automatic Light", "GPU", "bandwidth")],
        inplace=True,
        ascending=True,
    )
    write_performance_circle(
        aux_df,
        formats,
        ["CSR< Scalar >", "CSR< Vector >", "CSR< Light > Automatic Light"],
        "performance-graph-csr-1",
        scale,
        with_color_map=False,
    )
    write_performance_circle(
        aux_df,
        formats,
        ["CSR< Adaptive >", "CSR< Vector >", "CSR< Light > Automatic Light"],
        "performance-graph-csr-2",
        scale,
        with_color_map=False,
    )
    aux_df.sort_values(
        by=[("cusparse", "GPU", "bandwidth")], inplace=True, ascending=True
    )
    write_performance_circle(
        aux_df,
        formats,
        ["cusparse", "SlicedEllpack", "ChunkedEllpack"],
        "performance-graph-cusparse-ellpacks",
        scale,
        with_color_map=False,
    )
    write_performance_circle(
        aux_df,
        formats,
        ["cusparse", "CSR< Vector >", "CSR< Light > Automatic Light"],
        "performance-graph-cusparse-csr-1",
        scale,
        with_color_map=False,
    )
    write_performance_circle(
        aux_df,
        formats,
        ["cusparse", "CSR< Adaptive >", "CSR< Light > Automatic Light"],
        "performance-graph-cusparse-csr-2",
        scale,
        with_color_map=False,
    )
    write_performance_circle(
        aux_df,
        formats,
        ["cusparse", "CSR< Scalar >", "CSR< Light > Automatic Light"],
        "performance-graph-cusparse-csr-3",
        scale,
        with_color_map=False,
    )
    write_performance_circle(
        aux_df,
        formats,
        ["cusparse", "SlicedEllpack", "CSR< Light > Automatic Light"],
        "performance-graph-cusparse-csr-ellpack",
        scale,
        with_color_map=False,
    )
    with open("color-map.tex", "w") as file:
        write_colormap(file, 1200, 5, 13 * scale, 1.5 * scale, standalone=True)


####
# Make data analysis
def processDf(df, formats, head_size=10):
    print("Writting to HTML file...")
    df.sort_index(inplace=True)
    df.to_html(f"output.html")

    # Generate tables and figures
    effective_bw_profile(df, formats, head_size)
    cusparse_comparison(df, formats, head_size)
    csr_comparison(df, formats, head_size)
    legacy_formats_comparison(df, formats, head_size)
    csr_speedup_comparison(df, formats, head_size)
    # TODO: cusparse_and_hypre_speedup_comparison(df, formats, head_size)
    binary_matrices_comparison(df, formats, head_size)
    symmetric_matrices_comparison(df, formats, head_size)
    # TODO: csr_light_speedup_comparison(df, head_size)
    # csr_hypre_cpu_scalability(df, formats, head_size)
    # hypre_cpu_tnl_speedup_scalability(df, formats, head_size)

    best = df[("TNL Best", "GPU", "format", "")].tolist()
    best_formats = list(set(best))
    sum = 0
    for format in formats:
        if (
            not "Binary" in format
            and not "Symmetric" in format
            and not "Legacy" in format
            and not "LightSpMV" in format
            and not "TNL Best" in format
        ):
            cases = best.count(format)
            print(f"{format} is best in {cases} cases.")
            sum += cases
    print(f"Total is {sum}.")
    print(f"Best formats {best_formats}.")

    # write_performance_circles( df, formats )
    analyze_light_csr(df, formats)


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
print(f"Formats: {formats}")

threads_num_list = list(set(input_df["threads"].values.tolist()))
threads_num_list = [int(x) for x in threads_num_list if str(x) != "nan" and x != 0]
# if 0 in threads_num_list:
#    threads_num_list.remove(0)

accepted_formats = ["CSR"]
multicolumns, df_data = get_multiindex(
    input_df,
    formats,
    threads_num_list,
)
aux_df = pd.DataFrame(df_data, columns=multicolumns, index=[0])
aux_df.to_html("index.html")

print("Converting data...")
result = convert_data_frame(input_df, multicolumns, df_data, begin_idx=0, end_idx=-1)
result.to_html("sparse-matrix-benchmark-test-processed.html")

compute_speedup(result, formats, threads_num_list)
result.replace(to_replace=" ", value=np.nan, inplace=True)

print("Writting to file sparse-matrix-benchmark-test-processed.html ... ")
result.sort_index(inplace=True)
result.to_html("sparse-matrix-benchmark-test-processed.html")


head_size = 25
if not os.path.exists("general"):
    os.mkdir("general")
os.chdir("general")
processDf(result, formats, head_size)
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
