"""
Comparison of the SpMV on the CPU computed by TNL (the CSR format), Hypre and
Ginkgo with the same number of threads. Only these formats are benchmarked
with a fixed number of threads, the other TNL formats run only with the
default launch configuration.

For each number of threads p (the powers of two up to the largest number of
threads benchmarked) and each matrix, the time of each library is divided by
the best time of all libraries. The Dolan-More performance profile of a
library then shows the fraction of matrices on which the library is at most
tau-times slower than the best one, i.e. its value at tau = 1 is the fraction
of matrices on which the library is the fastest. The heatmap shows the number
of matrices on which each library is the fastest for each number of threads.

One more performance profile compares the libraries, each with the number of
threads which is the best for the given matrix and library, i.e. the time of
library L on matrix A is min_p t_L(A, p). This is a per-matrix oracle best,
not a configuration which can be chosen without knowing the results.

The parallel efficiency t_L(A, 1) / (p t_L(A, p)) of each library is shown by
its median over the matrices with the band between the 25th and the 75th
percentile, in one graph for all matrices and in another one for the matrices
grouped by the number of nonzeros (see NNZ_GROUPS). Only the matrices with the
results of all libraries for all numbers of threads are used.

Only the matrices with the results of all libraries for the given number of
threads are compared, the others are listed in a text file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from PosterGraphsCommon import (
    SEQUENTIAL_BLUE,
    INK_PRIMARY,
    INK_SECONDARY,
    GRIDLINE,
    _set_latex_rcparams,
)

# (format in the table, label in the graphs), the colors are the first three
# categorical slots, the line styles distinguish the libraries without colors
CPU_LIBRARIES = [("CSR", "TNL CSR"), ("Hypre", "Hypre"), ("Ginkgo", "Ginkgo")]
LIBRARY_STYLES = {
    "TNL CSR": {"color": "#2a78d6", "linestyle": "-"},
    "Hypre": {"color": "#eb6834", "linestyle": "--"},
    "Ginkgo": {"color": "#1baf7a", "linestyle": ":"},
}

MATRIX_NAME = ("Matrix name", "", "", "", "")
NONZEROS = ("nonzeros", "", "", "", "")

# (label, lower bound, upper bound) of the number of nonzeros, the labels give
# the approximate size of the CSR data in double precision (12 bytes per nonzero)
NNZ_GROUPS = [
    ("Small: $nnz < 10^5$ ($<$ 1.2 MB)", 0, 1e5),
    ("Medium: $10^5 \\leq nnz < 10^7$ (1.2--120 MB)", 1e5, 1e7),
    ("Large: $nnz \\geq 10^7$ ($>$ 120 MB)", 1e7, np.inf),
]


def _thread_count(launch_config):
    """
    Return the number of threads of a launch configuration like "1 thread" or
    "16 threads", or None for the other ones (e.g. "Default")
    """
    words = launch_config.split(" ")
    if len(words) == 2 and words[0].isdigit() and words[1] in ["thread", "threads"]:
        return int(words[0])
    return None


def _launch_config(threads):
    return "1 thread" if threads == 1 else f"{threads} threads"


def get_thread_counts(launch_configs):
    """
    Return the powers of two up to the largest number of threads benchmarked
    by any of the libraries
    """
    counts = [
        _thread_count(launch_config)
        for format, _ in CPU_LIBRARIES
        for launch_config in launch_configs.get((format, "CPU"), [])
    ]
    counts = [count for count in counts if count is not None]
    if not counts:
        return []
    return [2**i for i in range(int(np.log2(max(counts))) + 1)]


def collect_times(df, threads):
    """
    Return a dataframe indexed by the matrix name with the median time of each
    library for the given number of threads (NaN where it is missing)
    """
    times = pd.DataFrame(index=df[MATRIX_NAME].values)
    for format, label in CPU_LIBRARIES:
        column = (format, "CPU", _launch_config(threads), "time median", "")
        if column in df.columns:
            times[label] = pd.to_numeric(df[column], errors="coerce").values
        else:
            times[label] = np.nan
    return times


def performance_ratios(times):
    """
    Return the ratios of the times of each library to the best time on the
    matrices with the results of all libraries
    """
    complete = times.dropna()
    return complete.div(complete.min(axis=1), axis=0)


def best_thread_times(times_by_threads):
    """
    Return the best time of each library over all numbers of threads for each
    matrix and the number of threads giving it
    """
    stacked = pd.concat(times_by_threads, axis=1, names=["threads", "library"])
    best_times = stacked.T.groupby(level="library", sort=False).min().T
    best_threads = pd.DataFrame(index=stacked.index)
    for label in best_times.columns:
        times = stacked.xs(label, axis=1, level="library").dropna(how="all")
        best_threads[label] = times.idxmin(axis=1)
    return best_times, best_threads


def parallel_efficiency(times_by_threads):
    """
    Return the parallel efficiency t(1) / (p t(p)) of each library (columns
    (threads, library)) on the matrices with the results of all libraries for
    all numbers of threads
    """
    times = pd.concat(times_by_threads, axis=1, names=["threads", "library"]).dropna()
    efficiency = pd.DataFrame(index=times.index, columns=times.columns, dtype=float)
    for threads, label in times.columns:
        efficiency[(threads, label)] = times[(1, label)] / (threads * times[(threads, label)])
    return efficiency


def efficiency_statistics(efficiency):
    """
    Return the 25th percentile, the median and the 75th percentile of the
    efficiency over the matrices, indexed by (threads, library)
    """
    return efficiency.quantile([0.25, 0.5, 0.75]).T.set_axis(["q25", "median", "q75"], axis=1)


def write_missing_matrices(file_name, times_by_threads):
    """
    Write the matrices without the results of some library, with the numbers
    of threads for which they are missing
    """
    all_matrices = next(iter(times_by_threads.values())).index
    missing = {}
    for threads, times in times_by_threads.items():
        for label in times.columns:
            for matrix in times.index[times[label].isna()]:
                missing.setdefault((label, matrix), []).append(threads)

    summary = [f"Matrices in the table: {len(all_matrices)}"]
    for threads, times in times_by_threads.items():
        summary.append(
            f"   {_launch_config(threads)}: {len(times.dropna())} with the results of all libraries"
        )
    for label in LIBRARY_STYLES:
        count = len([matrix for library, matrix in missing if library == label])
        summary.append(f"   {label}: {count} matrices missing for some number of threads")
    print("\n".join(summary))

    with open(file_name, "w") as file:
        file.write("\n".join(summary) + "\n")
        for label in LIBRARY_STYLES:
            matrices = sorted(matrix for library, matrix in missing if library == label)
            if not matrices:
                continue
            file.write(f"\n{label}:\n")
            for matrix in matrices:
                threads = ", ".join(str(t) for t in missing[(label, matrix)])
                file.write(f"   missing {matrix:60} threads: {threads}\n")
    print(f"The matrices missing for some library are listed in {file_name}")


def draw_performance_profiles(
    ratios_by_panel,
    filename,
    title=None,
    panel_size=(4.2, 3.4),
    styles=LIBRARY_STYLES,
    tau_limit=None,
    tau_ticks=None,
):
    """
    Draw one panel with the performance profiles of all libraries for each
    item of ratios_by_panel (panel title -> ratios), the panels share both axes.
    The tau axis ends at the largest ratio or at tau_limit, if it is smaller,
    tau_ticks replace the ticks at the powers of ten.
    """
    _set_latex_rcparams()
    columns = min(3, len(ratios_by_panel))
    rows = int(np.ceil(len(ratios_by_panel) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    tau_max = max(ratios.values.max() for ratios in ratios_by_panel.values())

    for ax, (panel_title, ratios) in zip(axes.flat, ratios_by_panel.items()):
        n = len(ratios.index)
        for label in ratios.columns:
            tau = np.sort(ratios[label].values)
            fraction = np.arange(1, n + 1) / n
            # the profile starts at tau = 1 and stays constant up to tau_max
            tau = np.concatenate([[1.0], tau, [tau_max]])
            fraction = np.concatenate([[0.0], fraction, [1.0]])
            ax.step(tau, fraction, where="post", linewidth=2, label=label, **styles[label])
        ax.set_title(f"{panel_title} ($n={n}$)", color=INK_PRIMARY, fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(1, min(tau_max, tau_limit) if tau_limit else tau_max)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100 * y:.0f}\\%"))
        if tau_ticks:
            ax.set_xticks(tau_ticks)
            ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.tick_params(axis="both", length=0, colors=INK_PRIMARY)
        ax.grid(color=GRIDLINE, linewidth=1.0)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
    for ax in axes.flat[len(ratios_by_panel):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\tau$ (time / best time)", color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Matrices within $\tau$ of the best", color=INK_SECONDARY)

    # the panels may show different libraries
    legend = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(label, handle)
    fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=len(legend), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, color=INK_PRIMARY, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def draw_efficiency(efficiency_by_panel, filename, title=None, panel_size=(4.2, 3.4)):
    """
    Draw one panel for each item of efficiency_by_panel (panel title ->
    efficiency) with the median efficiency of each library and the band between
    the 25th and the 75th percentile, the panels share both axes
    """
    _set_latex_rcparams()
    columns = min(3, len(efficiency_by_panel))
    rows = int(np.ceil(len(efficiency_by_panel) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y_max = 1.1
    for ax, (panel_title, efficiency) in zip(axes.flat, efficiency_by_panel.items()):
        statistics = efficiency_statistics(efficiency)
        ax.axhline(1.0, color=INK_SECONDARY, linewidth=1.0, linestyle="-", zorder=1)
        for label in efficiency.columns.get_level_values("library").unique():
            library = statistics.xs(label, level="library")
            threads = library.index.values
            style = LIBRARY_STYLES[label]
            ax.fill_between(threads, library["q25"], library["q75"], color=style["color"], alpha=0.18,
                            linewidth=0, zorder=2)
            ax.plot(threads, library["median"], linewidth=2, marker="o", markersize=5, label=label,
                    zorder=3, **style)
            y_max = max(y_max, library["q75"].max() * 1.05)
        ax.set_title(f"{panel_title}, $n={len(efficiency.index)}$", color=INK_PRIMARY, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="both", length=0, colors=INK_PRIMARY)
        ax.grid(color=GRIDLINE, linewidth=1.0)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
    axes[0, 0].set_ylim(0, y_max)
    for ax in axes.flat[len(efficiency_by_panel):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Threads", color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Parallel efficiency", color=INK_SECONDARY)
    # the bottom panels of the columns without one in the last row need x labels too
    for ax in axes[-2, len(efficiency_by_panel) % columns:] if rows > 1 and len(efficiency_by_panel) % columns else []:
        ax.set_xlabel("Threads", color=INK_SECONDARY)
        ax.xaxis.set_tick_params(labelbottom=True)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, color=INK_PRIMARY, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def draw_wins_heatmap(ratios_by_threads, filename, title=None):
    """
    Draw the number (and the percentage) of matrices on which each library is
    the fastest for each number of threads
    """
    _set_latex_rcparams()
    threads_list = list(ratios_by_threads)
    labels = [label for _, label in CPU_LIBRARIES]
    wins = np.zeros((len(labels), len(threads_list)), dtype=int)
    totals = []
    for j, threads in enumerate(threads_list):
        ratios = ratios_by_threads[threads]
        totals.append(len(ratios.index))
        winners = ratios.idxmin(axis=1).value_counts()
        for i, label in enumerate(labels):
            wins[i, j] = winners.get(label, 0)
    shares = wins / np.maximum(np.array(totals), 1)

    fig, ax = plt.subplots(1, 1, figsize=(1.3 * len(threads_list) + 2, 0.8 * len(labels) + 1.2))
    colors = plt.matplotlib.colors.LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_BLUE)
    ax.imshow(shares, cmap=colors, vmin=0, vmax=1, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(threads_list)):
            ink = "white" if shares[i, j] > 0.55 else INK_PRIMARY
            ax.text(j, i, f"{wins[i, j]}\n{100 * shares[i, j]:.0f}\\%", ha="center", va="center",
                    fontsize=10, color=ink)
    ax.set_xticks(range(len(threads_list)))
    ax.set_xticklabels([str(threads) for threads in threads_list], color=INK_PRIMARY)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=INK_PRIMARY)
    ax.set_xlabel("Threads", color=INK_SECONDARY)
    ax.tick_params(axis="both", length=0)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, color=INK_PRIMARY, pad=10)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def cpu_libraries_graphs(df, launch_configs, transposed=False, output_dir="CPU-libraries", thread_counts=None):
    """
    Entry point: write the performance profiles and the heatmap of the wins of
    TNL CSR, Hypre and Ginkgo on the CPU, the ratios to the best time as CSV and
    the list of the matrices missing for some library.

    Only the given numbers of threads are compared (thread_counts), by default
    all of them (see get_thread_counts()). The parallel efficiency needs the
    results for one thread.
    """
    if thread_counts is None:
        thread_counts = get_thread_counts(launch_configs)
    if not thread_counts:
        print("No CPU results with a fixed number of threads, skipping the CPU libraries graphs.")
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    suffix = "-transposed" if transposed else ""
    matrices = "transposed matrices" if transposed else "original matrices"

    times_by_threads = {threads: collect_times(df, threads) for threads in thread_counts}
    write_missing_matrices(os.path.join(output_dir, f"missing-matrices{suffix}.txt"), times_by_threads)
    ratios_by_threads = {}
    for threads, times in times_by_threads.items():
        ratios = performance_ratios(times)
        if ratios.empty:
            print(f"No matrices with the results of all libraries for {_launch_config(threads)}.")
            continue
        ratios_by_threads[threads] = ratios
    if not ratios_by_threads:
        return

    csv = pd.concat(ratios_by_threads, axis=1, names=["threads", "library"])
    csv.index.name = "matrix"
    csv.to_csv(os.path.join(output_dir, f"ratios-to-best{suffix}.csv"))

    print(f"Writing the CPU libraries performance profiles for {len(ratios_by_threads)} numbers of threads")
    draw_performance_profiles(
        {_launch_config(threads): ratios for threads, ratios in ratios_by_threads.items()},
        os.path.join(output_dir, f"performance-profiles{suffix}.pdf"),
        title=f"Performance profiles of SpMV on the CPU, {matrices}",
    )
    draw_wins_heatmap(
        ratios_by_threads,
        os.path.join(output_dir, f"wins-heatmap{suffix}.pdf"),
        title=f"Fastest library on the CPU, {matrices}",
    )

    # the best number of threads for each matrix and library (per-matrix oracle)
    best_times, best_threads = best_thread_times(times_by_threads)
    best_ratios = performance_ratios(best_times)
    csv = pd.concat(
        {"ratio": best_ratios, "threads": best_threads.loc[best_ratios.index].astype(int)}, axis=1
    )
    csv.index.name = "matrix"
    csv.to_csv(os.path.join(output_dir, f"ratios-to-best-oracle{suffix}.csv"))
    print("Writing the CPU libraries performance profile with the best number of threads per matrix")
    draw_performance_profiles(
        {"Best number of threads per matrix (oracle)": best_ratios},
        os.path.join(output_dir, f"performance-profile-oracle{suffix}.pdf"),
        title=f"Performance profile of SpMV on the CPU, {matrices}",
        panel_size=(6.0, 4.2),
    )

    # the parallel efficiency for all matrices and for the groups by the number of nonzeros
    if 1 not in times_by_threads:
        print("The results for 1 thread are not included, skipping the parallel efficiency.")
        return
    efficiency = parallel_efficiency(times_by_threads)
    nonzeros = pd.Series(pd.to_numeric(df[NONZEROS], errors="coerce").values, index=df[MATRIX_NAME].values)
    nonzeros = nonzeros.loc[efficiency.index]
    efficiency_by_group = {}
    for label, lower, upper in NNZ_GROUPS:
        group = efficiency[(nonzeros >= lower) & (nonzeros < upper)]
        if not group.empty:
            efficiency_by_group[label] = group
    statistics = pd.concat(
        {
            panel: efficiency_statistics(group)
            for panel, group in {"All matrices": efficiency, **efficiency_by_group}.items()
        },
        names=["matrices"],
    )
    statistics.to_csv(os.path.join(output_dir, f"parallel-efficiency{suffix}.csv"))
    print("Writing the parallel efficiency of the CPU libraries")
    title = f"Parallel efficiency of SpMV on the CPU (median and 25th--75th percentile), {matrices}"
    draw_efficiency(
        {"All matrices": efficiency},
        os.path.join(output_dir, f"parallel-efficiency{suffix}.pdf"),
        title=title,
        panel_size=(6.0, 4.2),
    )
    if efficiency_by_group:
        draw_efficiency(
            efficiency_by_group,
            os.path.join(output_dir, f"parallel-efficiency-by-nonzeros{suffix}.pdf"),
            title=title,
        )
