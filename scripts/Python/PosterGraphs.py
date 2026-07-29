"""
Ad hoc, non-systematic graphs for the poster.

This module is a scratch space: functions here are added/removed/reshaped
while we figure out how the results should be presented. Nothing here is
expected to be stable API - once we settle on a presentation, it should move
into Report.py as a proper, systematic report.

Currently implemented:
  - speedup_overview_vs_cusparse: for every TNL format/launch-config on a
    given accelerator device, computes the percentage of matrices where the
    format is clearly faster than cuSPARSE, comparable (within a threshold),
    or clearly slower - and draws it as a 100% stacked horizontal bar chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

import LatexLabels

# Status palette (good / warning / critical) - fixed, never themed.
COLOR_FASTER = "#0ca30c"   # TNL clearly faster than the baseline
COLOR_SIMILAR = "#fab219"  # within the threshold, i.e. comparable performance
COLOR_SLOWER = "#d03b3b"   # baseline clearly faster than TNL

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def _tex_escape(text):
    return text.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def _set_latex_rcparams():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.size": 13,
        }
    )


def compute_speedup_breakdown(df, format, device, launch_config, reference="cusparse", threshold=0.10):
    """
    Compute the percentage of matrices for which `format` on `device` with
    `launch_config` is faster than / comparable to / slower than `reference`,
    based on the precomputed "speed-up" column (speed-up = reference_time /
    format_time, so speed-up > 1 means `format` is faster).

    Returns a dict with "faster", "similar", "slower" percentages (summing to
    100) and "n" (number of matrices the percentages are based on), or None
    if the column does not exist or has no data.
    """
    col = (format, device, launch_config, "speed-up", reference)
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    total = len(values)
    if total == 0:
        return None
    faster = int((values > 1 + threshold).sum())
    slower = int((values < 1 - threshold).sum())
    similar = total - faster - slower
    return {
        "faster": 100.0 * faster / total,
        "similar": 100.0 * similar / total,
        "slower": 100.0 * slower / total,
        "n": total,
    }


def default_label(format, launch_config):
    label = f"{format} {launch_config}".strip()
    return LatexLabels.latex_label(label)


def collect_speedup_overview(
    df,
    formats,
    launch_configs,
    device,
    reference="cusparse",
    threshold=0.10,
    skip_formats=("cusparse", "CSR Best"),
    exclude_substrings=("Binary", "Symmetric", "Sorted", "Legacy"),
    label_fn=None,
):
    """
    Build a list of (label, breakdown) tuples - one per (format, launch_config)
    combination available for the given accelerator device - suitable for
    draw_speedup_overview().

    By default, formats carrying a "Binary"/"Symmetric"/"Sorted"/"Legacy"
    variant tag are left out to keep the overview chart to one row per base
    format/launch-config; pass exclude_substrings=() to include everything.
    """
    if label_fn is None:
        label_fn = default_label
    rows = []
    for format in formats:
        if format in skip_formats:
            continue
        if any(substr in format for substr in exclude_substrings):
            continue
        if (format, device) not in launch_configs:
            continue
        for launch_config in launch_configs[(format, device)]:
            breakdown = compute_speedup_breakdown(
                df, format, device, launch_config, reference, threshold
            )
            if breakdown is None:
                continue
            rows.append((label_fn(format, launch_config), breakdown))
    return rows


def draw_speedup_overview(
    rows,
    filename,
    threshold=0.10,
    reference_label="cuSPARSE",
    title=None,
    fig_width=8,
    row_height=0.55,
):
    """
    Draw a 100% stacked horizontal bar chart from rows produced by
    collect_speedup_overview(): one row per format/launch-config, split into
    "TNL faster" / "within threshold" / "baseline faster" segments.
    """
    if not rows:
        print(f"No data to draw for {filename}, skipping.")
        return

    labels = [_tex_escape(r[0]) for r in rows]
    faster = np.array([r[1]["faster"] for r in rows])
    similar = np.array([r[1]["similar"] for r in rows])
    slower = np.array([r[1]["slower"] for r in rows])

    # rcParams must be set *before* the figure/text objects are created -
    # matplotlib bakes text.usetex into the text objects at creation time,
    # not at savefig() time.
    _set_latex_rcparams()

    y = np.arange(len(labels))
    fig_height = row_height * len(labels) + 1.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    pct = int(round(threshold * 100))
    segments = [
        (faster, COLOR_FASTER, "white"),
        (similar, COLOR_SIMILAR, INK_PRIMARY),
        (slower, COLOR_SLOWER, "white"),
    ]
    left = np.zeros(len(labels))
    for values, color, text_color in segments:
        ax.barh(
            y,
            values,
            left=left,
            height=0.62,
            color=color,
            edgecolor="white",
            linewidth=1.2,
        )
        for yi, (v, l) in enumerate(zip(values, left)):
            if v >= 8:
                ax.text(
                    l + v / 2,
                    yi,
                    f"{v:.0f}\\%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                )
        left = left + values

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}\\%"))
    ax.set_xlabel("Percentage of matrices in the test set", color=INK_SECONDARY)
    if title:
        ax.set_title(_tex_escape(title), color=INK_PRIMARY, pad=14)

    ax.tick_params(axis="both", length=0, colors=INK_PRIMARY)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLOR_FASTER, label=f"TNL faster ($>${pct}\\%)"),
        Patch(facecolor=COLOR_SIMILAR, label=f"within {pct}\\%"),
        Patch(facecolor=COLOR_SLOWER, label=f"{_tex_escape(reference_label)} faster"),
    ]
    # Legend goes below the x-axis label (rather than above the plot) so it
    # never competes with the title for the same vertical space.
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
    )

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def speedup_overview_vs_cusparse(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
):
    """
    Entry point: for every accelerator device, draw the "TNL vs cuSPARSE"
    overview chart with all available format/launch-config combinations.
    """
    if "cusparse" not in formats:
        print("No cusparse results found, skipping poster speedup overview.")
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_speedup_overview(
            df, formats, launch_configs, device, reference="cusparse", threshold=threshold
        )
        if not rows:
            continue
        print(f"Writing poster speedup overview for {device} ({len(rows)} configurations)")
        draw_speedup_overview(
            rows,
            filename=os.path.join(output_dir, f"speedup-vs-cusparse-{device}.pdf"),
            threshold=threshold,
            reference_label="cuSPARSE",
            title=f"TNL vs. cuSPARSE on {device}",
        )
