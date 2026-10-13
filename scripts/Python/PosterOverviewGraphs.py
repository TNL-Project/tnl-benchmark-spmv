"""
Poster/paper charts: 100%-stacked horizontal bar overviews of "how often is
TNL (or one TNL variant) faster/similar/slower than a reference", one row per
format/launch-config (or, for the sorted-segments chart, one row per
storage-family class). See PosterGraphsCommon.py for the shared color
palette and baseline-filter building blocks, PosterHeatmapGraphs.py for the
threshold heatmap, PosterCoverageGraphs.py for the cumulative-coverage
waterfall chart.

This module is a scratch space: functions here are added/removed/reshaped
while we figure out how the results should be presented. Nothing here is
expected to be stable API - once we settle on a presentation, it should move
into Report.py as a proper, systematic report.

Every comparison with a library stays on one device: CUDA results are
compared with cuSPARSE, HIP results with hipSPARSE (see VendorLibraries.py),
and both with Ginkgo/Hypre on that same device; the reference is the
library's best format/algorithm on each matrix.

Entry points:
  - speedup_overview_vs_libraries: for every TNL format/launch-config on a
    given accelerator device, computes the percentage of matrices where the
    format is clearly faster than the best time of a library (the vendor
    library of the device, Ginkgo, Hypre - see REFERENCE_LIBRARIES),
    comparable (within a threshold), or clearly slower - and draws it as a
    100% stacked horizontal bar chart, one chart per library.
    Alongside the PDF, also writes the same chart as a self-contained TikZ
    picture (write_speedup_overview_tikz) - a ".tex" file ready to \\input
    into a paper/poster (bar width/row height, and other layout offsets, are
    \\providecommand macros so they're retunable without regenerating the
    file), plus a "-standalone.tex" wrapper for quick previewing.
  - speedup_overview_csr_vs_libraries: the same charts (PDF + TikZ),
    restricted to CSR-family formats only - a narrower companion to
    speedup_overview_vs_libraries for when only the CSR kernels matter.
  - speedup_overview_sliced_ellpack_vs_vendor: the TNL SlicedEllpack
    variants (RowMajor/ColumnMajor, every slice size, each at its best
    launch config) against the vendor library's own Sliced ELLPACK (PDF +
    TikZ) - only on devices where the vendor library has one (cuSPARSE).
  - speedup_overview_variants: same breakdown, but one chart per
    "Binary"/"Symmetric"/"Sorted" variant tag, comparing each variant against
    its own non-variant counterpart (e.g. "Sorted CSR" vs "CSR") rather than
    against a library - these tags are left out of
    speedup_overview_vs_libraries to keep that chart to one row per base
    format/launch-config.
  - speedup_overview_csr_binary_vs_nonbinary: the Binary-vs-non-binary slice
    of speedup_overview_variants (PDF + TikZ), restricted to CSR-family
    formats only (i.e. "Binary CSR" vs "CSR") - the Binary/CSR-only
    counterpart to speedup_overview_csr_vs_libraries.
  - speedup_overview_ellpack_vs_libraries: the Ellpack counterpart of
    speedup_overview_csr_vs_libraries - the Ellpack variants of TNL (Ellpack,
    SlicedEllpack, BiEllpack, ChunkedEllpack) against the best result of each
    library, one row per format at its best launch config.
  - speedup_overview_ellpack_vs_best_csr: the same Ellpack variants against
    the best CSR of all libraries on the device and on the CPU.
  - speedup_heatmaps_variants_vs_libraries: the heatmaps of the three charts
    above for the Binary, Symmetric, Sorted and Sigma-256-Sorted variants of
    the formats, with the change w.r.t. the base format in each cell.
  - speedup_overview_sorted_segments: the impact of sorted segments (PDF +
    TikZ), one row per class (CSR, RowMajor/ColumnMajor SlicedEllpack)
    rather than one row per format/launch-config - each row compares the
    best time achievable with sorted segments against the best time
    achievable without, within that same class (see
    collect_sorted_segments_overview() / SORTED_SEGMENTS_CLASSES).

Every bar chart (PDF) comes with a heatmap of the same rows in the directory
GPU-heatmaps (see HEATMAPS_DIR), in the file with the suffix "-margins": the percentage of matrices where the subject, resp.
the reference, is faster by more than 25%, 50%, 100%, 200% and 400% (see
SPEEDUP_MARGINS / draw_speedup_margins_heatmap()), and with the same heatmap
with the numbers of matrices in the file with the suffix "-margins-counts".
The charts comparing TNL with the best result of another library use smaller
margins (10%, 25%, 50% and 100%, see LIBRARY_SPEEDUP_MARGINS), since the
library chooses its best format and algorithm for each matrix, so large
speed-ups of one TNL format are rare. These charts are drawn for the
transposed matrices too, to the files with the suffix "-transposed".
"""

import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

import LatexLabels
from VendorLibraries import is_vendor_format, vendor_format, vendor_label
from PosterGraphsCommon import (
    COLOR_FASTER,
    COLOR_SIMILAR,
    COLOR_SLOWER,
    INK_PRIMARY,
    INK_SECONDARY,
    GRIDLINE,
    _tex_escape,
    _set_latex_rcparams,
    _write_tikz_standalone_wrapper,
    _min_time_per_matrix,
    FAMILY_FILTERS,
    _is_csr_baseline_format,
    _best_vendor_time,
    _vendor_format_filter,
)

# ---------------------------------------------------------------------------
# Speed-up breakdown: classifying a series of speed-up values into
# "faster" / "similar" / "slower" percentages. Every chart in this module
# (overview bar charts, sorted-segments chart) reduces to this one formula,
# so it lives in a single place - check the math here once, trust it
# everywhere else.
# ---------------------------------------------------------------------------


# Margins of the heatmap drawn next to every overview bar chart (see
# draw_speedup_margins_heatmap()): 0.25 stands for "faster by more than 25%".
SPEEDUP_MARGINS = (0.25, 0.50, 1.00, 2.00, 4.00)
# Margins of the charts comparing TNL with the best result of another library.
LIBRARY_SPEEDUP_MARGINS = (0.10, 0.25, 0.50, 1.00)
# Directory of all heatmaps of the margins, next to the directory of the charts.
HEATMAPS_DIR = "GPU-heatmaps"
# The texts in the cells of the heatmaps have the size of the labels around
# (see _set_latex_rcparams()). The cells drawn without LaTeX use the Computer
# Modern Sans font shipped with matplotlib, which is the font of the labels.
HEATMAP_CELL_FONT_SIZE = 13
HEATMAP_CELL_FONT = "cmss10"
ALL_SPEEDUP_MARGINS = sorted(set(SPEEDUP_MARGINS) | set(LIBRARY_SPEEDUP_MARGINS))


def _speedup_breakdown(speedup_values, threshold):
    """
    Classify a Series of speed-up values (convention: speed-up > 1 means the
    subject is faster than the reference) into three buckets:
      - "faster": speed-up > 1 + threshold
      - "slower": speed-up < 1 - threshold
      - "similar": everything else, i.e. within +/- threshold of 1
    and express each bucket as a percentage of the (non-NaN) total.

    Returns a dict with "faster"/"similar"/"slower" percentages (summing to
    100) and "n" (the number of matrices behind them), or None if there are
    no valid values at all. For draw_speedup_margins_heatmap(), the dict also
    holds "faster by" and "slower by": the numbers of matrices where the
    subject, resp. the reference, is faster by more than each margin of
    ALL_SPEEDUP_MARGINS (margin -> count), i.e. where speed-up > 1 + margin,
    resp. 1 / speed-up > 1 + margin.
    """
    values = speedup_values.dropna()
    total = len(values)
    if total == 0:
        return None
    faster = int((values > 1 + threshold).sum())
    slower = int((values < 1 - threshold).sum())
    similar = total - faster - slower  # whatever is left, by construction
    return {
        "faster": 100.0 * faster / total,
        "similar": 100.0 * similar / total,
        "slower": 100.0 * slower / total,
        "n": total,
        "faster by": {m: int((values > 1 + m).sum()) for m in ALL_SPEEDUP_MARGINS},
        "slower by": {m: int((1 / values > 1 + m).sum()) for m in ALL_SPEEDUP_MARGINS},
    }


def compute_speedup_breakdown(df, format, device, launch_config, reference, threshold=0.10):
    """
    Compute the percentage of matrices for which `format` on `device` with
    `launch_config` is faster than / comparable to / slower than `reference`,
    based on the precomputed "speed-up" column (speed-up = reference_time /
    format_time, so speed-up > 1 means `format` is faster) - see
    _speedup_breakdown() for the faster/similar/slower classification itself.

    Returns None if the column does not exist or has no data.
    """
    col = (format, device, launch_config, "speed-up", reference)
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    return _speedup_breakdown(values, threshold)


def compute_speedup_breakdown_vs_time(df, format, device, launch_config, reference_time, threshold=0.10):
    """
    Same as compute_speedup_breakdown(), but against a per-matrix reference
    time Series (e.g. the best vendor library time, see _best_vendor_time())
    instead of a precomputed "speed-up" column: speed-up = reference_time /
    format_time.

    Returns None if the format has no "time median" column or no data.
    """
    col = (format, device, launch_config, "time median", "")
    if col not in df.columns:
        return None
    values = reference_time / pd.to_numeric(df[col], errors="coerce")
    return _speedup_breakdown(values, threshold)


# Formats of the third-party SpMV libraries benchmarked alongside TNL.
THIRD_PARTY_FORMAT_PREFIXES = ("Ginkgo", "Hypre", "LightSpMV", "CSR5", "Petsc")


def _is_library_format(format):
    """True for vendor and third-party library formats, i.e. anything but TNL."""
    return is_vendor_format(format) or format.startswith(THIRD_PARTY_FORMAT_PREFIXES)


# Tags of the variants of the formats: a format without any of them is a base
# format, the name of its variant is "<tag> <base format>".
VARIANT_NAME_TAGS = ("Binary", "Symmetric", "Sorted", "Legacy")


def _is_pure_variant(format, tag):
    """
    True for the variant `tag` of a base format alone, e.g. "Sorted CSR" for
    the tag "Sorted", but not "Binary Sorted CSR", "Symmetric Sorted CSR" or
    "Sigma-256-Sorted CSR", which combine it with other variants.
    """
    prefix = f"{tag} "
    return format.startswith(prefix) and not any(other in format[len(prefix):] for other in VARIANT_NAME_TAGS)


def default_label(format, launch_config):
    label = f"{format} {launch_config}".strip()
    return LatexLabels.latex_label(label)


# ---------------------------------------------------------------------------
# Overview bar charts: one 100%-stacked bar per row, each row's
# faster/similar/slower breakdown coming from _speedup_breakdown() above.
# Rows are usually one per (format, launch_config); speedup_overview_sorted_
# segments() is the exception, with one row per storage-family class instead
# (see collect_sorted_segments_overview()).
# ---------------------------------------------------------------------------


def collect_speedup_overview(
    df,
    formats,
    launch_configs,
    device,
    reference=None,
    reference_time=None,
    threshold=0.10,
    skip_formats=("CSR Best",),
    exclude_substrings=("Binary", "Symmetric", "Sorted", "Legacy"),
    require_substring=None,
    variant_tag=None,
    format_filter=None,
    label_fn=None,
):
    """
    Build a list of (label, breakdown) tuples - one per (format, launch_config)
    combination available for the given accelerator device - suitable for
    draw_speedup_overview().

    Each row is compared either with the precomputed "speed-up" column vs.
    `reference` (e.g. "non-binary"), or - when reference_time is given -
    with that per-matrix reference time (e.g. the best vendor library time).
    Only TNL formats get a row: the vendor libraries and the third-party
    libraries (THIRD_PARTY_FORMAT_PREFIXES) are always skipped.

    By default, formats carrying a "Binary"/"Symmetric"/"Sorted"/"Legacy"
    variant tag are left out to keep the overview chart to one row per base
    format/launch-config; pass exclude_substrings=() to include everything.

    If require_substring is given (e.g. "Sorted"), exclude_substrings is
    ignored and only formats containing that substring are kept - use this to
    build a separate overview for one variant tag (see
    speedup_overview_variants()).

    If variant_tag is given (e.g. "Sorted"), only the pure variants of the
    base formats are kept, i.e. "<tag> <base format>" where the base format
    carries no variant tag itself (see _is_pure_variant()), and
    exclude_substrings and require_substring are ignored.

    format_filter, if given, is an extra predicate applied on top of the
    above (e.g. restrict to CSR-family formats only, see
    speedup_overview_csr_vs_libraries()).
    """
    if label_fn is None:
        label_fn = default_label
    rows = []
    for format in formats:
        if format in skip_formats or _is_library_format(format):
            continue
        if variant_tag is not None:
            if not _is_pure_variant(format, variant_tag):
                continue
        elif require_substring is not None:
            if require_substring not in format:
                continue
        elif any(substr in format for substr in exclude_substrings):
            continue
        if format_filter is not None and not format_filter(format):
            continue
        if (format, device) not in launch_configs:
            continue
        for launch_config in launch_configs[(format, device)]:
            if reference_time is not None:
                breakdown = compute_speedup_breakdown_vs_time(
                    df, format, device, launch_config, reference_time, threshold
                )
            else:
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
    subject_label="TNL",
    reference_label="reference",
    title=None,
    fig_width=8,
    row_height=0.55,
    margins=SPEEDUP_MARGINS,
    heatmaps_dir=HEATMAPS_DIR,
):
    """
    Draw a 100% stacked horizontal bar chart from rows produced by
    collect_speedup_overview(): one row per format/launch-config, split into
    "subject_label faster" / "within threshold" / "reference_label faster"
    segments. The bars tell only who is faster by more than the threshold, so
    the heatmap of the same rows with more margins is written to the
    directory heatmaps_dir, to the file with the suffix "-margins", and the
    same heatmap with the numbers of matrices to the file with the suffix
    "-margins-counts" (see draw_speedup_margins_heatmap()).
    """
    if not rows:
        print(f"No data to draw for {filename}, skipping.")
        return
    base, extension = os.path.splitext(os.path.basename(filename))
    os.makedirs(heatmaps_dir, exist_ok=True)
    for suffix, counts in [("-margins", False), ("-margins-counts", True)]:
        draw_speedup_margins_heatmap(
            rows,
            os.path.join(heatmaps_dir, f"{base}{suffix}{extension}"),
            subject_label=subject_label,
            reference_label=reference_label,
            title=title,
            margins=margins,
            counts=counts,
        )

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
        Patch(facecolor=COLOR_FASTER, label=f"{_tex_escape(subject_label)} faster ($>${pct}\\%)"),
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


def draw_speedup_margins_heatmap(
    rows,
    filename,
    subject_label="TNL",
    reference_label="reference",
    title=None,
    cell_width=0.95,
    row_height=0.42,
    margins=SPEEDUP_MARGINS,
    counts=False,
    base_breakdowns=None,
):
    """
    Draw a heatmap from the rows of draw_speedup_overview(): one row per
    format/launch-config and one column per margin on each side. The cells on
    the left show the percentage of matrices where the reference is faster by
    more than the margin (red), the cells on the right where the subject is
    faster by more than the margin (green), with the smallest margin next to
    the middle. The percentages are cumulative: the matrices of a larger
    margin are counted in the smaller ones too. "Faster by 100%" means twice
    as fast on both sides.

    With counts=True, the cells show the numbers of matrices instead of the
    percentages (the colors are the same) and the number of all compared
    matrices is written next to each row.

    base_breakdowns, if given, holds the breakdown of the base format of each
    row computed on the same matrices (or None). The change of each cell
    w.r.t. the base format is then written in parentheses: in percentage
    points, with counts=True as the number of matrices.
    """
    margins = list(margins)
    totals = np.array([row[1]["n"] for row in rows], dtype=float)[:, None]
    slower_counts = np.array([[row[1]["slower by"][m] for m in reversed(margins)] for row in rows])
    faster_counts = np.array([[row[1]["faster by"][m] for m in margins] for row in rows])
    slower = 100.0 * slower_counts / totals
    faster = 100.0 * faster_counts / totals
    labels = [_tex_escape(row[0]) for row in rows]

    _set_latex_rcparams()
    columns = 2 * len(margins)
    fig, ax = plt.subplots(
        1, 1, figsize=(cell_width * columns + 3.0, row_height * len(rows) + 1.8)
    )
    colormap = plt.matplotlib.colors.LinearSegmentedColormap.from_list
    extent_slower = (-0.5, len(margins) - 0.5, len(rows) - 0.5, -0.5)
    extent_faster = (len(margins) - 0.5, columns - 0.5, len(rows) - 0.5, -0.5)
    ax.imshow(slower, cmap=colormap("slower", ["white", COLOR_SLOWER]), vmin=0, vmax=100,
              aspect="auto", extent=extent_slower)
    ax.imshow(faster, cmap=colormap("faster", ["white", COLOR_FASTER]), vmin=0, vmax=100,
              aspect="auto", extent=extent_faster)
    # The numbers of matrices and the changes make thousands of different
    # texts and LaTeX is run for each new one, which takes minutes. Such cells
    # are drawn without LaTeX, the percentages alone repeat and are cached.
    cells_usetex = not (counts or base_breakdowns)
    percent = "\\%" if cells_usetex else "%"
    cell_font = {"fontsize": HEATMAP_CELL_FONT_SIZE}
    if not cells_usetex:
        cell_font["fontfamily"] = HEATMAP_CELL_FONT
    for i in range(len(rows)):
        row_counts = np.concatenate([slower_counts[i], faster_counts[i]])
        base = base_breakdowns[i] if base_breakdowns else None
        if base is not None:
            base_counts = np.array(
                [base["slower by"][m] for m in reversed(margins)] + [base["faster by"][m] for m in margins]
            )
            base_values = 100.0 * base_counts / base["n"]
        for j, value in enumerate(np.concatenate([slower[i], faster[i]])):
            text = f"{row_counts[j]}" if counts else f"{value:.0f}{percent}"
            if base is not None:
                if counts:
                    text += f" ({row_counts[j] - base_counts[j]:+d})"
                else:
                    text += f" ({int(round(value - base_values[j])):+d})"
            ax.text(j, i, text, ha="center", va="center", color="white" if value > 60 else INK_PRIMARY,
                    usetex=cells_usetex, **cell_font)
        if counts:
            ax.text(columns - 0.4, i, f"of {totals[i, 0]:.0f}", ha="left", va="center",
                    color=INK_SECONDARY, usetex=cells_usetex, **cell_font)
    ax.axvline(len(margins) - 0.5, color=INK_PRIMARY, linewidth=1.2)
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)

    ax.set_xticks(range(columns))
    ax.set_xticklabels(
        [f"$>${100 * m:.0f}\\%" for m in reversed(margins)] + [f"$>${100 * m:.0f}\\%" for m in margins],
        color=INK_PRIMARY,
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, color=INK_PRIMARY)
    ax.tick_params(axis="both", length=0)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    # the sides are described under the margins
    for position, label in [(0.25, reference_label), (0.75, subject_label)]:
        ax.annotate(f"{_tex_escape(label)} faster by more than", xy=(position, 0), xycoords="axes fraction",
                    xytext=(0, -26), textcoords="offset points", ha="center", va="top", color=INK_SECONDARY)
    if title:
        ax.set_title(_tex_escape(title), color=INK_PRIMARY, pad=10)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# TikZ export for the overview bar charts (write_speedup_overview_tikz).
# Coordinates below are all in "data units": x in [0, 1] spans one full row
# (0%-100%), y in [0, 1] spans one row's height - the tikzpicture's own
# [x=..., y=...] options (set from \SpeedupOverviewBarWidth/RowHeight) then
# scale those units to real cm, so the drawing logic here never has to think
# in physical lengths.
# ---------------------------------------------------------------------------

# Gives each write_speedup_overview_tikz() call a distinct TikZ node-name
# prefix for its legend, so \input-ing several of these charts into the same
# document never collides (TikZ node names are global, not per-picture).
_tikz_legend_counter = itertools.count()


def write_speedup_overview_tikz(
    rows,
    filename,
    threshold=0.10,
    subject_label="TNL",
    reference_label="reference",
    title=None,
    bar_width=12.0,
    row_height=1.25,
    wrap_reference_label=False,
    standalone=True,
):
    """
    Write the 100% stacked horizontal bar chart from collect_speedup_overview()
    as a self-contained TikZ picture - same colors/percentages as
    draw_speedup_overview(), except there is no "Percentage of matrices..."
    axis caption (the 0/20/.../100% tick labels are considered enough) and
    the legend sits directly below the bars, laid out in one row rather than
    stacked - so it can be \\input into a paper or poster.

    wrap_reference_label breaks the reference legend item ("<reference_label>
    faster") onto two lines (via \\shortstack) instead of one - useful when
    reference_label itself is a compound word like "non-binary" that would
    otherwise make that one legend item much wider than the other two.

    Row width (the length standing for 100%) and row height are emitted as
    \\providecommand macros - \\SpeedupOverviewBarWidth and
    \\SpeedupOverviewRowHeight - so they (and the other layout offsets) can be
    retuned directly in the .tex, or overridden from the surrounding document
    with \\renewcommand before the first \\input, without regenerating the
    file. `bar_width`/`row_height` (cm) only set their initial defaults. Note
    \\providecommand only takes effect once per document: if several of these
    overview charts are \\input together, the first one's macros win unless
    overridden earlier.

    `filename` should end in ".tex" and receives just the tikzpicture, ready
    to \\input. When standalone is True, a second file - same name with a
    "-standalone" suffix - is also written, wrapping the picture in a
    compilable standalone document for quick previewing.
    """
    if not rows:
        print(f"No data to draw for {filename}, skipping.")
        return

    n_rows = len(rows)
    pct = int(round(threshold * 100))

    # Symbolic names of the \providecommand macros below - used instead of
    # their numeric values in the coordinate expressions further down, so the
    # emitted .tex stays retunable (see the docstring above).
    m_row_label_gap = "\\SpeedupOverviewRowLabelGap"
    m_tick_label_y = "\\SpeedupOverviewTickLabelYOffset"
    m_legend_x = "\\SpeedupOverviewLegendXOffset"
    m_legend_y = "\\SpeedupOverviewLegendYOffset"
    m_legend_item_gap = "\\SpeedupOverviewLegendItemGap"
    m_legend_swatch_gap = "\\SpeedupOverviewLegendSwatchGap"
    m_title_y_gap = "\\SpeedupOverviewTitleYGap"

    lines = []
    lines.append(f"\\definecolor{{inkprimary}}{{HTML}}{{{INK_PRIMARY.lstrip('#')}}}")
    lines.append(f"\\definecolor{{colorfaster}}{{HTML}}{{{COLOR_FASTER.lstrip('#')}}}")
    lines.append(f"\\definecolor{{colorsimilar}}{{HTML}}{{{COLOR_SIMILAR.lstrip('#')}}}")
    lines.append(f"\\definecolor{{colorslower}}{{HTML}}{{{COLOR_SLOWER.lstrip('#')}}}")
    lines.append(f"\\providecommand{{\\SpeedupOverviewBarWidth}}{{{bar_width}cm}}")
    lines.append(f"\\providecommand{{\\SpeedupOverviewRowHeight}}{{{row_height}cm}}")
    lines.append(f"\\providecommand{{{m_row_label_gap}}}{{-0.02}}")
    lines.append(f"\\providecommand{{{m_tick_label_y}}}{{-0.35}}")
    lines.append(f"\\providecommand{{{m_legend_x}}}{{0.0}}")
    lines.append(f"\\providecommand{{{m_legend_y}}}{{-1.8}}")
    lines.append(f"\\providecommand{{{m_legend_item_gap}}}{{0.5cm}}")
    lines.append(f"\\providecommand{{{m_legend_swatch_gap}}}{{0.1cm}}")
    lines.append(f"\\providecommand{{{m_title_y_gap}}}{{0.5}}")
    lines.append(
        "\\begin{tikzpicture}[x=\\SpeedupOverviewBarWidth, y=\\SpeedupOverviewRowHeight]"
    )

    # Bars sit inset within their row (bar_inset on each side, in y-units),
    # leaving a visible gap between rows, like matplotlib's barh height=0.62.
    bar_inset = 0.19
    for yi, (label, breakdown) in enumerate(rows):
        # rows[0] is drawn at the top: yi=0 -> y=n_rows-1 (TikZ y grows
        # upward), yi=n_rows-1 -> y=0.
        y = n_rows - 1 - yi
        left = 0.0  # running x-position (in [0, 1]) as segments are placed
        for value, colorname, text_color in (
            (breakdown["faster"], "colorfaster", "white"),
            (breakdown["similar"], "colorsimilar", "inkprimary"),
            (breakdown["slower"], "colorslower", "white"),
        ):
            width = value / 100.0  # percentage -> fraction of the full row
            if width > 0:
                lines.append(
                    f"  \\fill[{colorname}] ({left:.4f},{y + bar_inset:.4f}) rectangle "
                    f"++({width:.4f},{1 - 2 * bar_inset:.4f});"
                )
                if value >= 8:
                    lines.append(
                        f"  \\node[text={text_color}] at "
                        f"({left + width / 2:.4f},{y + 0.5:.4f}) {{{value:.0f}\\%}};"
                    )
            left += width
        lines.append(
            f"  \\node[anchor=east, text=inkprimary] at ({m_row_label_gap},{y + 0.5:.4f}) "
            f"{{{_tex_escape(label)}}};"
        )

    for tick in range(0, 101, 20):
        x = tick / 100.0
        lines.append(
            f"  \\node[anchor=north, text=inkprimary] at ({x:.2f},{m_tick_label_y}) {{{tick}\\%}};"
        )

    # Laid out side by side using TikZ's own relative positioning (each
    # item placed a real, absolute xshift after the previous one's actual
    # rendered width) rather than pre-computed slot widths, so items never
    # overlap regardless of how long subject_label/reference_label make the
    # text, or how narrow bar_width is.
    if wrap_reference_label:
        reference_text = f"\\shortstack[l]{{{_tex_escape(reference_label)}\\\\faster}}"
    else:
        reference_text = f"{_tex_escape(reference_label)} faster"
    legend_items = [
        ("colorfaster", f"{_tex_escape(subject_label)} faster ($>${pct}\\%)"),
        ("colorsimilar", f"within {pct}\\%"),
        ("colorslower", reference_text),
    ]
    legend_prefix = f"speedupOverviewLegend{next(_tikz_legend_counter)}"
    prev_text_name = None
    for i, (colorname, text) in enumerate(legend_items):
        swatch_name = f"{legend_prefix}sw{i}"
        text_name = f"{legend_prefix}tx{i}"
        if prev_text_name is None:
            pos = f"({m_legend_x},{m_legend_y})"
        else:
            pos = f"([xshift={m_legend_item_gap}]{prev_text_name}.east)"
        lines.append(
            f"  \\node[fill={colorname}, minimum width=0.3cm, minimum height=0.3cm, "
            f"inner sep=0] ({swatch_name}) at {pos} {{}};"
        )
        lines.append(
            f"  \\node[anchor=west, text=inkprimary] ({text_name}) at "
            f"([xshift={m_legend_swatch_gap}]{swatch_name}.east) {{{text}}};"
        )
        prev_text_name = text_name

    if title:
        lines.append(
            f"  \\node[anchor=south, text=inkprimary, font=\\bfseries] "
            f"at (0.5,{n_rows}+{m_title_y_gap}) {{{_tex_escape(title)}}};"
        )

    lines.append("\\end{tikzpicture}")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    if standalone:
        _write_tikz_standalone_wrapper(filename)


def _is_plain_csr_format(format):
    return "CSR" in format


def _is_plain_binary_csr_format(format):
    # Used with require_substring="Binary" (which bypasses the default
    # exclude_substrings filtering in collect_speedup_overview), so Symmetric
    # and Sorted have to be excluded here explicitly to keep just "Binary
    # CSR" and out "Sorted Binary CSR" / "Symmetric Binary CSR" etc.
    return "CSR" in format and not any(tag in format for tag in ("Symmetric", "Sorted"))


def _label_without_binary(format, launch_config):
    # "Binary" is already implied by the chart's subject_label/title, so
    # dropping it from each row's own label avoids repeating it on every row.
    stripped = format.replace("Binary", "").strip()
    return default_label(stripped, launch_config)


# Libraries the "TNL vs. <library>" overview charts compare with, on every
# accelerator device where the library has results: "vendor" stands for the
# vendor library of the device (cuSPARSE on CUDA, hipSPARSE on HIP), the
# others are the third-party formats of the same name.
REFERENCE_LIBRARIES = ("vendor", "Ginkgo", "Hypre")


def _transposed_suffix(transposed):
    return "-transposed" if transposed else ""


def _transposed_title(transposed):
    return ", transposed matrices" if transposed else ""


def _reference_library(df, device, library):
    """
    (label, per-matrix best time) of `library` (see REFERENCE_LIBRARIES) on
    `device` - its best format/algorithm/launch config on each matrix - or
    None if the library has no results on that device.
    """
    if library == "vendor":
        label = vendor_label(device)
        reference_time = _best_vendor_time(df, device)
    else:
        label = library
        reference_time = _min_time_per_matrix(df, device, lambda format: format == library)
    if label is None or reference_time is None:
        return None
    return label, reference_time


def _speedup_overview_vs_libraries(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    libraries,
    output_dir,
    threshold,
    subject_label,
    format_filter=None,
    filename_tag="",
    transposed=False,
):
    """
    Shared implementation of speedup_overview_vs_libraries() and
    speedup_overview_csr_vs_libraries(): for every accelerator device and
    every library in `libraries` with results on that device, one overview
    chart (PDF + TikZ) of the TNL formats passing `format_filter` against
    the library's best time on each matrix. The charts for the transposed
    matrices have "-transposed" in the file name and say so in the title.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for library in libraries:
        for device in accelerator_devices:
            reference = _reference_library(df, device, library)
            if reference is None:
                print(f"No {library} results found on {device}, skipping poster speedup overview.")
                continue
            label, reference_time = reference
            rows = collect_speedup_overview(
                df,
                formats,
                launch_configs,
                device,
                reference_time=reference_time,
                threshold=threshold,
                format_filter=format_filter,
            )
            if not rows:
                continue
            print(
                f"Writing poster {subject_label} speedup overview vs. {label} for {device} "
                f"({len(rows)} configurations)"
            )
            basename = f"speedup{filename_tag}-vs-{label.lower()}-{device}{_transposed_suffix(transposed)}"
            title = f"{subject_label} vs. best {label} on {device}{_transposed_title(transposed)}"
            draw_speedup_overview(
                rows,
                filename=os.path.join(output_dir, f"{basename}.pdf"),
                threshold=threshold,
                subject_label=subject_label,
                reference_label=label,
                title=title,
                margins=LIBRARY_SPEEDUP_MARGINS,
            )
            write_speedup_overview_tikz(
                rows,
                filename=os.path.join(output_dir, f"{basename}.tex"),
                threshold=threshold,
                subject_label=subject_label,
                reference_label=label,
                title=title,
            )


def speedup_overview_vs_libraries(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    libraries=REFERENCE_LIBRARIES,
    output_dir="Poster",
    threshold=0.10,
    transposed=False,
):
    """
    Entry point: for every accelerator device and every library in
    `libraries` (default: the vendor library of the device, Ginkgo and
    Hypre), draw the "TNL vs. best <library>" overview chart with all
    available TNL format/launch-config combinations - the reference on each
    matrix is the library's best result on that same device. Alongside the
    PDF, also writes the same chart as a self-contained TikZ picture
    (write_speedup_overview_tikz) - a ".tex" file ready to \\input into a
    paper/poster, plus a "-standalone.tex" wrapper for previewing.
    """
    _speedup_overview_vs_libraries(
        df,
        formats,
        launch_configs,
        accelerator_devices,
        libraries,
        output_dir,
        threshold,
        subject_label="TNL",
        transposed=transposed,
    )


def speedup_overview_csr_vs_libraries(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    libraries=REFERENCE_LIBRARIES,
    output_dir="Poster",
    threshold=0.10,
    transposed=False,
):
    """
    Entry point: for every accelerator device and library, draw a narrower
    companion to speedup_overview_vs_libraries() restricted to CSR-family
    formats only (still excluding Binary/Symmetric/Sorted/Legacy variants,
    same as the main overview) - PDF plus the same TikZ output.
    """
    _speedup_overview_vs_libraries(
        df,
        formats,
        launch_configs,
        accelerator_devices,
        libraries,
        output_dir,
        threshold,
        subject_label="CSR",
        format_filter=_is_plain_csr_format,
        filename_tag="-csr",
        transposed=transposed,
    )


def _sliced_ellpack_classes():
    """
    Rows of the TNL SlicedEllpack vs. vendor Sliced ELLPACK chart, as
    (label, format filter) pairs: the best of all plain (untagged) TNL
    SlicedEllpack formats, the best per element organization, then every
    organization/slice size on its own - each at its best launch config.
    """
    classes = [
        (
            "Best SlicedEllpack",
            lambda format: FAMILY_FILTERS["RowMajor SlicedEllpack"](format)
            or FAMILY_FILTERS["ColumnMajor SlicedEllpack"](format),
        ),
    ]
    for organization in ("RowMajor", "ColumnMajor"):
        family = f"{organization} SlicedEllpack"
        classes.append((f"Best {family}", FAMILY_FILTERS[family]))
    for organization in ("RowMajor", "ColumnMajor"):
        for slice_size in (2, 4, 8, 16, 32):
            name = f"{organization} SlicedEllpack {slice_size}"
            classes.append((name, lambda format, name=name: format == name))
    return classes


def collect_sliced_ellpack_vs_vendor_sell(df, device, threshold=0.10):
    """
    Compare each class from _sliced_ellpack_classes() (best launch config,
    and for the "Best ..." rows also best slice size/organization) with the
    vendor library's best Sliced ELLPACK time (best algorithm) on the same
    device, using the "speed-up = reference/subject, >1 means subject
    faster" convention of compute_speedup_breakdown().

    Returns a list of (label, breakdown) tuples suitable for
    draw_speedup_overview() / write_speedup_overview_tikz(), or [] if the
    vendor library has no Sliced ELLPACK results on this device.
    """
    vendor_ok = _vendor_format_filter(device)
    if vendor_ok is None:
        return []
    vendor_sell_time = _min_time_per_matrix(
        df, device, lambda format: vendor_ok(format) and "SlicedEll" in format
    )
    if vendor_sell_time is None:
        return []
    rows = []
    for label, format_ok in _sliced_ellpack_classes():
        tnl_time = _min_time_per_matrix(df, device, format_ok)
        if tnl_time is None:
            continue
        breakdown = _speedup_breakdown(vendor_sell_time / tnl_time, threshold)
        if breakdown is None:
            continue
        rows.append((label, breakdown))
    return rows


def speedup_overview_sliced_ellpack_vs_vendor(
    df,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
    transposed=False,
):
    """
    Entry point: for every accelerator device whose vendor library offers a
    Sliced ELLPACK format (currently cuSPARSE only), draw the TNL
    SlicedEllpack variants against it (see
    collect_sliced_ellpack_vs_vendor_sell()) - PDF plus the TikZ output
    (write_speedup_overview_tikz).
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_sliced_ellpack_vs_vendor_sell(df, device, threshold=threshold)
        if not rows:
            continue
        vendor = vendor_label(device)
        print(
            f"Writing poster SlicedEllpack overview vs. {vendor} SlicedEll for {device} "
            f"({len(rows)} rows)"
        )
        reference_label = f"{vendor} SlicedEll"
        title = f"TNL SlicedEllpack vs. best {vendor} SlicedEll on {device}{_transposed_title(transposed)}"
        basename = (
            f"speedup-slicedellpack-vs-{vendor.lower()}-slicedell-{device}{_transposed_suffix(transposed)}"
        )
        draw_speedup_overview(
            rows,
            filename=os.path.join(output_dir, f"{basename}.pdf"),
            threshold=threshold,
            subject_label="TNL",
            reference_label=reference_label,
            title=title,
            margins=LIBRARY_SPEEDUP_MARGINS,
        )
        write_speedup_overview_tikz(
            rows,
            filename=os.path.join(output_dir, f"{basename}.tex"),
            threshold=threshold,
            subject_label="TNL",
            reference_label=reference_label,
            title=title,
            wrap_reference_label=True,
        )


def _ellpack_classes():
    """
    Rows of the TNL Ellpack variants vs. library charts, as (label, format
    filter) pairs: the best of all plain (untagged) Ellpack-like formats, each
    family (Ellpack, RowMajor/ColumnMajor SlicedEllpack, BiEllpack,
    ChunkedEllpack) and every SlicedEllpack organization/slice size on its
    own - each at its best launch config.
    """
    classes = [("Best Ellpack variant", lambda format: any(ok(format) for ok in FAMILY_FILTERS.values()))]
    for family, format_ok in FAMILY_FILTERS.items():
        label = f"Best {family}" if "SlicedEllpack" in family else family
        classes.append((label, format_ok))
    for organization in ("RowMajor", "ColumnMajor"):
        for slice_size in (2, 4, 8, 16, 32):
            name = f"{organization} SlicedEllpack {slice_size}"
            classes.append((name, lambda format, name=name: format == name))
    return classes


def collect_ellpack_vs_library(df, device, reference_time, threshold=0.10):
    """
    Compare each class from _ellpack_classes() (its best launch config, and
    for the "Best ..." rows also the best format of the class, on each matrix)
    with the per-matrix reference time of a library. Returns a list of (label,
    breakdown) tuples suitable for draw_speedup_overview() /
    write_speedup_overview_tikz().
    """
    rows = []
    for label, format_ok in _ellpack_classes():
        tnl_time = _min_time_per_matrix(df, device, format_ok)
        if tnl_time is None:
            continue
        breakdown = _speedup_breakdown(reference_time / tnl_time, threshold)
        if breakdown is None:
            continue
        rows.append((label, breakdown))
    return rows


def speedup_overview_ellpack_vs_libraries(
    df,
    accelerator_devices,
    libraries=REFERENCE_LIBRARIES,
    output_dir="Poster",
    threshold=0.10,
    transposed=False,
):
    """
    Entry point: the Ellpack counterpart of speedup_overview_csr_vs_libraries()
    - for every accelerator device and library, the Ellpack variants of TNL
    (see _ellpack_classes()) against the library's best time on each matrix,
    as PDF plus the TikZ output. Unlike the CSR chart, there is one row per
    format with its best launch config on each matrix, not one row per launch
    config, which would make more than 60 rows.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for library in libraries:
        for device in accelerator_devices:
            reference = _reference_library(df, device, library)
            if reference is None:
                print(f"No {library} results found on {device}, skipping poster Ellpack overview.")
                continue
            label, reference_time = reference
            rows = collect_ellpack_vs_library(df, device, reference_time, threshold=threshold)
            if not rows:
                continue
            print(f"Writing poster Ellpack overview vs. {label} for {device} ({len(rows)} rows)")
            basename = f"speedup-ellpack-vs-{label.lower()}-{device}{_transposed_suffix(transposed)}"
            title = f"Ellpack variants vs. best {label} on {device}{_transposed_title(transposed)}"
            draw_speedup_overview(
                rows,
                filename=os.path.join(output_dir, f"{basename}.pdf"),
                threshold=threshold,
                subject_label="TNL",
                reference_label=label,
                title=title,
                margins=LIBRARY_SPEEDUP_MARGINS,
            )
            write_speedup_overview_tikz(
                rows,
                filename=os.path.join(output_dir, f"{basename}.tex"),
                threshold=threshold,
                subject_label="TNL",
                reference_label=label,
                title=title,
            )


# CSR implementations of the other libraries: they are benchmarked under the
# name of the library (the vendor library under vendor_format(device)).
CSR_LIBRARY_FORMAT_PREFIXES = ("Ginkgo", "Hypre", "LightSpMV")


def _is_any_csr_format(device):
    """
    Predicate matching every CSR implementation benchmarked on `device`: the
    plain CSR formats of TNL (CSR and Adaptive CSR, also with sorted rows, see
    _is_csr_baseline_format()), the CSR format of the vendor library of the
    device (not its other formats) and the libraries which offer only CSR.
    """
    vendor_csr = vendor_format(device)

    def format_ok(format):
        if format == vendor_csr or format.startswith(CSR_LIBRARY_FORMAT_PREFIXES):
            return True
        return _is_csr_baseline_format(format) and not _is_library_format(format)

    return format_ok


def _best_csr_time(df, device):
    """
    Per-matrix best time of any CSR implementation of any library, on `device`
    or on the CPU with any number of threads ("best CSR ever"), and the sorted
    list of the formats it is taken over as (format, device) pairs. Returns
    None if there is no such result.
    """
    candidates = []
    used = []
    for performer in (device, "CPU"):
        format_ok = _is_any_csr_format(performer)
        time = _min_time_per_matrix(df, performer, format_ok)
        if time is None:
            continue
        candidates.append(time)
        used += sorted(
            {
                (column[0], performer)
                for column in df.columns
                if len(column) == 5 and column[1] == performer and column[3] == "time median" and format_ok(column[0])
            }
        )
    if not candidates:
        return None
    return pd.concat(candidates, axis=1).min(axis=1), used


def speedup_overview_ellpack_vs_best_csr(
    df,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
    transposed=False,
):
    """
    Entry point: for every accelerator device, the Ellpack variants of TNL
    (see _ellpack_classes()) against the best CSR of all libraries on that
    device and on the CPU (see _best_csr_time()) - PDF plus the TikZ output.
    The formats the best CSR is taken over are written to a text file.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        best_csr = _best_csr_time(df, device)
        if best_csr is None:
            continue
        reference_time, used = best_csr
        rows = collect_ellpack_vs_library(df, device, reference_time, threshold=threshold)
        if not rows:
            continue
        print(f"Writing poster Ellpack overview vs. best CSR for {device} ({len(rows)} rows)")
        basename = f"speedup-ellpack-vs-best-csr-{device}{_transposed_suffix(transposed)}"
        title = f"Ellpack variants vs. best CSR of all libraries on {device} and CPU{_transposed_title(transposed)}"
        with open(os.path.join(output_dir, f"{basename}-formats.txt"), "w") as file:
            file.write("The best CSR is the best time of these formats with all their launch configurations:\n")
            for format, performer in used:
                file.write(f"   {format} on {performer}\n")
        draw_speedup_overview(
            rows,
            filename=os.path.join(output_dir, f"{basename}.pdf"),
            threshold=threshold,
            subject_label="TNL",
            reference_label="best CSR",
            title=title,
            margins=LIBRARY_SPEEDUP_MARGINS,
        )
        write_speedup_overview_tikz(
            rows,
            filename=os.path.join(output_dir, f"{basename}.tex"),
            threshold=threshold,
            subject_label="TNL",
            reference_label="best CSR",
            title=title,
        )


# ---------------------------------------------------------------------------
# Heatmaps of the variants of the formats (Binary, Symmetric, Sorted,
# Sigma-256-Sorted) vs. the other libraries: the same rows and references as
# the CSR and Ellpack charts above, with the change w.r.t. the base format.
# ---------------------------------------------------------------------------

# Variants of the formats: the name of the variant is "<tag> <base format>".
VARIANT_TAGS = ("Binary", "Symmetric", "Sorted", "Sigma-256-Sorted")


def _variant_and_base_breakdown(reference_time, variant_time, base_time, threshold):
    """
    Breakdowns of the speed-ups of a variant and of its base format w.r.t. the
    reference on the same matrices - those with the results of all three - so
    that they can be compared. Without the base format, the variant is
    evaluated on all its matrices and the base breakdown is None.
    """
    if variant_time is None:
        return None, None
    if base_time is None:
        return _speedup_breakdown(reference_time / variant_time, threshold), None
    common = reference_time.notna() & variant_time.notna() & base_time.notna()
    return (
        _speedup_breakdown((reference_time / variant_time)[common], threshold),
        _speedup_breakdown((reference_time / base_time)[common], threshold),
    )


def _column_time(df, format, device, launch_config):
    column = (format, device, launch_config, "time median", "")
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def collect_csr_variant_vs_library(df, formats, launch_configs, device, tag, reference_time, threshold=0.10):
    """
    Rows of the CSR chart (see speedup_overview_csr_vs_libraries()) for the
    variant `tag` of the CSR formats, one per launch config, as (label,
    breakdown, breakdown of the base format) tuples.
    """
    rows = []
    for base in formats:
        if _is_library_format(base) or not _is_csr_baseline_format(base) or "Sorted" in base:
            continue
        format = f"{tag} {base}"
        for launch_config in launch_configs.get((format, device), []):
            breakdown, base_breakdown = _variant_and_base_breakdown(
                reference_time,
                _column_time(df, format, device, launch_config),
                _column_time(df, base, device, launch_config),
                threshold,
            )
            if breakdown is not None:
                rows.append((default_label(format, launch_config), breakdown, base_breakdown))
    return rows


def collect_ellpack_variant_vs_library(df, device, tag, reference_time, threshold=0.10):
    """
    Rows of the Ellpack charts (see _ellpack_classes()) for the variant `tag`
    of the Ellpack formats, as (label, breakdown, breakdown of the base
    class) tuples.
    """
    prefix = f"{tag} "
    rows = []
    for label, format_ok in _ellpack_classes():
        variant_ok = lambda format, format_ok=format_ok: format.startswith(prefix) and format_ok(format[len(prefix):])
        breakdown, base_breakdown = _variant_and_base_breakdown(
            reference_time,
            _min_time_per_matrix(df, device, variant_ok),
            _min_time_per_matrix(df, device, format_ok),
            threshold,
        )
        if breakdown is not None:
            rows.append((f"{tag} {label}" if not label.startswith("Best") else f"Best {tag} {label[5:]}", breakdown, base_breakdown))
    return rows


def _draw_variant_heatmaps(rows, basename, output_dir, title, reference_label):
    """
    Write the heatmaps of the margins with the percentages and with the
    numbers of matrices for the rows of a variant
    """
    if not rows:
        return
    for suffix, counts in [("-margins", False), ("-margins-counts", True)]:
        draw_speedup_margins_heatmap(
            [(label, breakdown) for label, breakdown, _ in rows],
            os.path.join(output_dir, f"{basename}{suffix}.pdf"),
            subject_label="TNL",
            reference_label=reference_label,
            title=title,
            cell_width=1.45,
            margins=LIBRARY_SPEEDUP_MARGINS,
            counts=counts,
            base_breakdowns=[base for _, _, base in rows],
        )


def speedup_heatmaps_variants_vs_libraries(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    libraries=REFERENCE_LIBRARIES,
    output_dir=HEATMAPS_DIR,
    threshold=0.10,
    transposed=False,
):
    """
    Entry point: the heatmaps of the charts speedup-csr-vs-<library>,
    speedup-ellpack-vs-<library> and speedup-ellpack-vs-best-csr for the
    Binary, Symmetric, Sorted and Sigma-256-Sorted variants of the formats
    (see VARIANT_TAGS), i.e. each variant against the same reference as its
    base format. Each cell shows also the change w.r.t. the base format in
    parentheses - in percentage points, resp. in the number of matrices in
    the "-margins-counts" files. The variant and its base format are
    evaluated on the same matrices, those with the results of both, since
    e.g. the symmetric formats are benchmarked only on symmetric matrices.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    suffix = _transposed_suffix(transposed)
    for device in accelerator_devices:
        references = []
        for library in libraries:
            reference = _reference_library(df, device, library)
            if reference is not None:
                references.append((reference[0].lower(), f"best {reference[0]}", reference[0], reference[1]))
        best_csr = _best_csr_time(df, device)
        for tag in VARIANT_TAGS:
            # the Sigma-256-Sorted formats are compared with the non-sorted ones too
            base_name = "non-sorted" if tag.endswith("Sorted") else f"non-{tag.lower()}"
            change = f"change w.r.t. the {base_name} formats in parentheses"
            print(f"Writing poster heatmaps of the {tag} formats vs. the libraries for {device}")
            for name, reference_title, reference_label, reference_time in references:
                _draw_variant_heatmaps(
                    collect_csr_variant_vs_library(
                        df, formats, launch_configs, device, tag, reference_time, threshold
                    ),
                    f"speedup-{tag.lower()}-csr-vs-{name}-{device}{suffix}",
                    output_dir,
                    f"{tag} CSR vs. {reference_title} on {device}{_transposed_title(transposed)} ({change})",
                    reference_label,
                )
                _draw_variant_heatmaps(
                    collect_ellpack_variant_vs_library(df, device, tag, reference_time, threshold),
                    f"speedup-{tag.lower()}-ellpack-vs-{name}-{device}{suffix}",
                    output_dir,
                    f"{tag} Ellpack variants vs. {reference_title} on {device}"
                    f"{_transposed_title(transposed)} ({change})",
                    reference_label,
                )
            if best_csr is not None:
                _draw_variant_heatmaps(
                    collect_ellpack_variant_vs_library(df, device, tag, best_csr[0], threshold),
                    f"speedup-{tag.lower()}-ellpack-vs-best-csr-{device}{suffix}",
                    output_dir,
                    f"{tag} Ellpack variants vs. best CSR of all libraries on {device} and CPU"
                    f"{_transposed_title(transposed)} ({change})",
                    "best CSR",
                )


def speedup_overview_variants(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
    variant_classes=("Symmetric", "Binary", "Sorted", "Sigma-256-Sorted"),
):
    """
    Entry point: for every accelerator device and every variant tag in
    variant_classes, draw an overview chart restricted to the pure variants
    of the base formats with that tag (e.g. "Sorted CSR", but not "Binary
    Sorted CSR", see _is_pure_variant()), comparing each one against its own
    non-variant counterpart (e.g. "Sorted CSR" vs "CSR") using the "speed-up"
    vs. "non-binary"/"non-symmetric"/"non-sorted" column that
    tnl-spmv-benchmark-parse-logs.py already computes for these
    formats.

    These formats are excluded from speedup_overview_vs_libraries() to keep
    the main overview to one row per base format/launch-config, so without
    this function they would never show up in the poster output at all.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for variant_class in variant_classes:
        # the Sigma-256-Sorted formats are compared with the non-sorted ones too
        reference = "non-sorted" if variant_class.endswith("Sorted") else f"non-{variant_class.lower()}"
        for device in accelerator_devices:
            rows = collect_speedup_overview(
                df,
                formats,
                launch_configs,
                device,
                reference=reference,
                threshold=threshold,
                variant_tag=variant_class,
            )
            if not rows:
                continue
            print(
                f"Writing poster speedup overview for {variant_class} vs. {reference} "
                f"on {device} ({len(rows)} configurations)"
            )
            draw_speedup_overview(
                rows,
                filename=os.path.join(
                    output_dir, f"speedup-{variant_class.lower()}-vs-{reference}-{device}.pdf"
                ),
                threshold=threshold,
                subject_label=variant_class,
                reference_label=reference,
                title=f"{variant_class} vs. {reference} formats on {device}",
            )


def speedup_overview_csr_binary_vs_nonbinary(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
):
    """
    Entry point: for every accelerator device, draw a narrower companion to
    speedup_overview_variants()'s Binary-vs-non-binary chart, restricted to
    plain CSR only (i.e. "Binary CSR" vs "CSR" - Symmetric and Sorted variants
    are excluded, unlike speedup_overview_csr_vs_libraries() which only needs
    to exclude Binary/Symmetric/Sorted/Legacy indirectly via the default
    exclude_substrings) - PDF plus the same TikZ output
    (write_speedup_overview_tikz).
    """
    reference = "non-binary"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_speedup_overview(
            df,
            formats,
            launch_configs,
            device,
            reference=reference,
            threshold=threshold,
            require_substring="Binary",
            format_filter=_is_plain_binary_csr_format,
            label_fn=_label_without_binary,
        )
        if not rows:
            continue
        print(
            f"Writing poster CSR speedup overview for Binary vs. {reference} "
            f"on {device} ({len(rows)} configurations)"
        )
        draw_speedup_overview(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-csr-binary-vs-{reference}-{device}.pdf"
            ),
            threshold=threshold,
            subject_label="Binary CSR",
            reference_label=reference,
            title=f"Binary CSR vs. {reference} CSR on {device}",
        )
        write_speedup_overview_tikz(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-csr-binary-vs-{reference}-{device}.tex"
            ),
            threshold=threshold,
            subject_label="Binary CSR",
            reference_label=reference,
            title=f"Binary CSR vs. {reference} CSR on {device}",
            wrap_reference_label=True,
        )


# "Sorted segments" comparison classes: for each class, a (sorted_filter,
# nonsorted_filter) pair used by collect_sorted_segments_overview() to find
# the best time achievable within each side. RowMajor/ColumnMajor
# SlicedEllpack's sorted variants are named "Sorted RowMajor SlicedEllpack
# <n>" (tag prefixed, like "Symmetric Sorted ..."), so matching on
# startswith("Sorted ...") already excludes the Symmetric/Binary-tagged
# variants; the CSR class reuses _is_csr_baseline_format (CSR + Adaptive CSR,
# no Binary/Symmetric/Legacy/Best) and just splits it further on "Sorted".
SORTED_SEGMENTS_CLASSES = {
    "CSR": (
        lambda format: _is_csr_baseline_format(format) and "Sorted" in format,
        lambda format: _is_csr_baseline_format(format) and "Sorted" not in format,
    ),
    "RowMajor SlicedEllpack": (
        lambda format: format.startswith("Sorted RowMajor SlicedEllpack"),
        FAMILY_FILTERS["RowMajor SlicedEllpack"],
    ),
    "ColumnMajor SlicedEllpack": (
        lambda format: format.startswith("Sorted ColumnMajor SlicedEllpack"),
        FAMILY_FILTERS["ColumnMajor SlicedEllpack"],
    ),
}


def collect_sorted_segments_overview(df, device, classes=None, threshold=0.10):
    """
    For every class in `classes` (CSR, RowMajor/ColumnMajor SlicedEllpack),
    compare the best time achievable with sorted segments against the best
    time achievable without (both the best across launch configs and, for
    CSR, across {CSR, Adaptive CSR}), using the same "speed-up =
    reference/subject, >1 means subject faster" convention as
    compute_speedup_breakdown().

    Returns a list of (label, breakdown) tuples - same shape as
    collect_speedup_overview() - suitable for draw_speedup_overview() /
    write_speedup_overview_tikz(), skipping classes with no data for this
    device (either side).
    """
    if classes is None:
        classes = SORTED_SEGMENTS_CLASSES
    rows = []
    for label, (sorted_filter, nonsorted_filter) in classes.items():
        sorted_time = _min_time_per_matrix(df, device, sorted_filter)
        nonsorted_time = _min_time_per_matrix(df, device, nonsorted_filter)
        if sorted_time is None or nonsorted_time is None:
            continue
        # subject = sorted, reference = non-sorted -> speed-up > 1 means
        # sorting made this class faster on this matrix.
        speedup = nonsorted_time / sorted_time
        breakdown = _speedup_breakdown(speedup, threshold)
        if breakdown is None:
            continue
        rows.append((label, breakdown))
    return rows


def speedup_overview_sorted_segments(
    df,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
):
    """
    Entry point: for every accelerator device, draw a chart showing the
    impact of sorted segments - one row per class (CSR, RowMajor
    SlicedEllpack, ColumnMajor SlicedEllpack), each comparing the best time
    achievable with sorted segments against the best time achievable without,
    within that same class - PDF plus the TikZ output
    (write_speedup_overview_tikz).
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_sorted_segments_overview(df, device, threshold=threshold)
        if not rows:
            continue
        print(
            f"Writing poster sorted-segments speedup overview for {device} "
            f"({len(rows)} classes)"
        )
        draw_speedup_overview(
            rows,
            filename=os.path.join(output_dir, f"speedup-sorted-segments-{device}.pdf"),
            threshold=threshold,
            subject_label="Sorted",
            reference_label="non-sorted",
            title=f"Impact of sorted segments on {device}",
        )
        write_speedup_overview_tikz(
            rows,
            filename=os.path.join(output_dir, f"speedup-sorted-segments-{device}.tex"),
            threshold=threshold,
            subject_label="Sorted",
            reference_label="non-sorted",
            title=f"Impact of sorted segments on {device}",
            wrap_reference_label=True,
        )
