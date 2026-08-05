"""
Low-level helpers shared by the poster/paper chart modules
(PosterOverviewGraphs.py, PosterHeatmapGraphs.py, PosterCoverageGraphs.py).

Nothing in here draws a chart on its own - it's the color palette, small
LaTeX/TikZ plumbing, and the "best time per matrix" / "plain CSR baseline"
building blocks that more than one of those modules needs. Keeping them here
(rather than duplicated, or hanging off whichever module happened to need
them first) is what lets the three chart modules stay independent of each
other while still agreeing on colors and on what "the CSR baseline" means.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Status palette (good / warning / critical) - fixed, never themed.
COLOR_FASTER = "#0ca30c"   # TNL clearly faster than the baseline
COLOR_SIMILAR = "#fab219"  # within the threshold, i.e. comparable performance
COLOR_SLOWER = "#d03b3b"   # baseline clearly faster than TNL

# Sequential (single-hue, light -> dark) palette for magnitude/heatmap encoding.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

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


def _write_tikz_standalone_wrapper(filename):
    """
    Write "<base>-standalone.tex" next to `filename` (which must contain just
    a tikzpicture, no preamble): a minimal compilable document that \\input's
    it, so a single chart can be previewed on its own (e.g. `pdflatex
    foo-standalone.tex`) without assembling the full paper/poster around it.
    """
    base, ext = os.path.splitext(filename)
    standalone_filename = f"{base}-standalone{ext}"
    with open(standalone_filename, "w") as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\begin{document}\n")
        f.write(f"\\input{{{os.path.basename(filename)}}}\n")
        f.write("\\end{document}\n")


def _min_time_per_matrix(df, device, format_ok):
    """
    Per-matrix minimum of "time mean" across all (format, launch_config)
    columns on `device` whose format passes `format_ok` - i.e. the best time
    achievable on that matrix by picking the best launch config (and, when
    format_ok matches several formats, the best format) from the group.

    Returns None if no matching column exists for this device.
    """
    cols = [
        col
        for col in df.columns
        if len(col) == 5 and col[1] == device and col[3] == "time mean" and format_ok(col[0])
    ]
    if not cols:
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").min(axis=1)


# ---------------------------------------------------------------------------
# "Best plain CSR" baseline: the reference every chart module compares
# against, in one place so the definition of "plain CSR" (and the set of
# storage families) can't drift between modules.
# ---------------------------------------------------------------------------

# "Plain" CSR variants (no Binary/Symmetric compression, which change the
# problem/data assumptions) - these form the "best CSR" half of the baseline
# used by PosterHeatmapGraphs.speedup_heatmap_vs_best_csr() and
# PosterCoverageGraphs.cumulative_coverage_vs_best(): CSR, Adaptive CSR,
# Sorted CSR, Sorted Adaptive CSR and their launch configs.
CSR_BASELINE_EXCLUDE_TAGS = ("Binary", "Symmetric", "Legacy", "Best")

# One row/step per storage family. RowMajor/ColumnMajor SlicedEllpack fold
# their slice-size suffix (e.g. "RowMajor SlicedEllpack 16") into one entry
# via best-of-family selection; the plain formats are matched exactly so they
# don't also swallow BiEllpack/ChunkedEllpack, which contain "Ellpack" too.
FAMILY_FILTERS = {
    "Ellpack": lambda format: format == "Ellpack",
    "RowMajor SlicedEllpack": lambda format: format.startswith("RowMajor SlicedEllpack"),
    "ColumnMajor SlicedEllpack": lambda format: format.startswith("ColumnMajor SlicedEllpack"),
    "BiEllpack": lambda format: format == "BiEllpack",
    "ChunkedEllpack": lambda format: format == "ChunkedEllpack",
}


def _is_csr_baseline_format(format):
    return "CSR" in format and not any(tag in format for tag in CSR_BASELINE_EXCLUDE_TAGS)
