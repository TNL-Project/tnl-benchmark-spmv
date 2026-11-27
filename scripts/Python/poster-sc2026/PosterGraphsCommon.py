"""
Low-level helpers shared by the poster/paper chart modules
(PosterOverviewGraphs.py, PosterHeatmapGraphs.py, PosterCoverageGraphs.py).

Nothing in here draws a chart on its own - it's the color palette, small
LaTeX/TikZ plumbing, and the "best time per matrix" / "plain CSR baseline" /
"vendor library" building blocks (defined in FormatFilters.py, shared with the
analysis tables) that more than one of those modules needs. Keeping them here
(rather than duplicated, or hanging off whichever module happened to need
them first) is what lets the three chart modules stay independent of each
other while still agreeing on colors, on what "the CSR baseline" means and
on which vendor library each device is compared with.
"""

import os
import matplotlib.pyplot as plt

# the selection of the formats is shared with the analysis tables, it is
# re-exported here for the chart modules
from FormatFilters import (  # noqa: F401
    _min_time_per_matrix,
    CSR_BASELINE_EXCLUDE_TAGS,
    FAMILY_FILTERS,
    _is_csr_baseline_format,
    _vendor_format_filter,
    GINKGO_VENDOR_FORMAT,
    _library_format_filter,
    _best_vendor_time,
)

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
