"""
Poster/paper charts: the "which TNL storage family beats the best CSR /
cuSPARSE baseline, and by how much" heatmap, plus the per-matrix Markdown
report that names the individual matrices behind it. See
PosterGraphsCommon.py for the shared color palette and baseline-filter
building blocks, PosterOverviewGraphs.py for the row/bar overview charts,
PosterCoverageGraphs.py for the cumulative-coverage waterfall chart.

This module is a scratch space: functions here are added/removed/reshaped
while we figure out how the results should be presented. Nothing here is
expected to be stable API - once we settle on a presentation, it should move
into Report.py as a proper, systematic report.

Entry points:
  - speedup_heatmap_vs_best_csr: for every accelerator device, a heatmap with
    one row per "storage family" (Ellpack, RowMajor/ColumnMajor SlicedEllpack,
    BiEllpack, ChunkedEllpack - each taken at its best launch config and, for
    SlicedEllpack, best slice size) and one column per speedup threshold,
    showing the percentage of matrices where the family beats the best of
    (plain CSR, cuSPARSE) by more than that threshold. Alongside the PDF,
    speedup_heatmap_vs_best_csr also writes the same heatmap as a
    self-contained TikZ picture (write_speedup_heatmap_tikz) - a ".tex" file
    ready to \\input into a paper/poster, plus a "-standalone.tex" wrapper
    for quick previewing.
  - write_speedup_matrices: for every accelerator device, a Markdown file
    naming the individual matrices where some storage family beats the best
    CSR / cuSPARSE baseline by at least a given speedup threshold (default
    1.1x) - grouped by format (families with the highest speedups first),
    sorted by descending speedup within each format. Each matrix also lists
    its per-matrix statistics (MATRIX_STAT_COLUMNS, as computed by
    MatrixStatistics.h / emitted by spmv.h), alongside the min/max/avg of
    each statistic over every matrix in the dataset, for context.

  Both speedup_heatmap_vs_best_csr and write_speedup_matrices take an
  include_cpu_csr flag: by default the "best CSR" half of the baseline is
  restricted to CSR runs on the same accelerator device, but with
  include_cpu_csr=True the best plain CSR time on the CPU is also considered
  (a TNL user willing to fall back to the CPU would take it if it beats
  everything on the accelerator) - a separate, "-incl-cpu-csr"-suffixed set
  of outputs is written for this variant.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from PosterGraphsCommon import (
    SEQUENTIAL_BLUE,
    INK_PRIMARY,
    INK_SECONDARY,
    _tex_escape,
    _set_latex_rcparams,
    _write_tikz_standalone_wrapper,
    _min_time_per_matrix,
    FAMILY_FILTERS,
    _is_csr_baseline_format,
)

# ---------------------------------------------------------------------------
# Heatmap vs. best CSR / cuSPARSE baseline: for each storage family, what
# fraction of matrices beat the best available "sane default" reference
# (plain CSR and/or cuSPARSE, optionally also plain CSR on the CPU) by more
# than each of DEFAULT_SPEEDUP_THRESHOLDS. Shared with the per-matrix
# Markdown report (write_speedup_matrices) further below, which names the
# individual matrices behind these percentages.
# ---------------------------------------------------------------------------

# Speedup axis for the heatmap: 1.00x, 1.05x, ..., 1.50x.
DEFAULT_SPEEDUP_THRESHOLDS = [round(1.0 + 0.05 * i, 2) for i in range(11)]


def _best_csr_cusparse_baseline_time(
    df, device, csr_filter=_is_csr_baseline_format, include_cpu_csr=False
):
    """
    Per-matrix best (lowest) time among: the best "plain" CSR variant on
    `device` (see CSR_BASELINE_EXCLUDE_TAGS in PosterGraphsCommon.py),
    cuSPARSE on `device`, and - when include_cpu_csr is True - also the best
    plain CSR variant on the CPU (i.e. a TNL user willing to fall back to a
    CPU run would take that if it beats everything available on `device`).
    Returns None if none of these are available for this device.
    """
    candidates = [
        _min_time_per_matrix(df, device, csr_filter),
        _min_time_per_matrix(df, device, lambda format: format == "cusparse"),
    ]
    if include_cpu_csr:
        candidates.append(_min_time_per_matrix(df, "CPU", csr_filter))
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return pd.concat(candidates, axis=1).min(axis=1)


def collect_speedup_heatmap_vs_best_csr(
    df,
    device,
    family_filters=None,
    csr_filter=_is_csr_baseline_format,
    thresholds=DEFAULT_SPEEDUP_THRESHOLDS,
    include_cpu_csr=False,
):
    """
    Build heatmap rows for one device: for every storage family in
    family_filters, the percentage of matrices for which the family's best
    time (best launch config, and best format within the family - e.g. best
    slice size for SlicedEllpack) beats the baseline by more than each
    speedup threshold in `thresholds` (e.g. 1.05 means "more than 5% faster").

    The baseline for a matrix is whichever is faster: the best "plain" CSR
    variant on `device` (see CSR_BASELINE_EXCLUDE_TAGS in
    PosterGraphsCommon.py), cuSPARSE on `device`, and - when include_cpu_csr
    is True - also the best plain CSR variant on the CPU - i.e. the best
    reference a TNL user could actually reach for on that matrix.

    Returns a list of (label, percentages, counts, n) tuples, skipping
    families with no data for this device. `counts` are the raw number of
    matrices behind each `percentages` entry (percentages[i] == 100 *
    counts[i] / n); draw_speedup_heatmap() can display either.
    """
    if family_filters is None:
        family_filters = FAMILY_FILTERS

    baseline_time = _best_csr_cusparse_baseline_time(
        df, device, csr_filter=csr_filter, include_cpu_csr=include_cpu_csr
    )
    if baseline_time is None:
        return []

    thresholds = list(thresholds)
    rows = []
    for label, family_ok in family_filters.items():
        family_time = _min_time_per_matrix(df, device, family_ok)
        if family_time is None:
            continue
        # subject = family, reference = baseline -> speed-up > 1 means the
        # family beats the baseline on this matrix (same "reference/subject"
        # convention used across the poster chart modules), but thresholded
        # per-column below instead of collapsed into a single
        # faster/similar/slower breakdown.
        speedup = baseline_time / family_time
        valid = speedup.notna()
        total = int(valid.sum())
        if total == 0:
            continue
        counts = [int((speedup[valid] > t).sum()) for t in thresholds]
        percentages = [100.0 * c / total for c in counts]
        rows.append((label, percentages, counts, total))
    return rows


def draw_speedup_heatmap(
    rows,
    filename,
    thresholds=DEFAULT_SPEEDUP_THRESHOLDS,
    title=None,
    fig_width=8,
    row_height=0.6,
    value_kind="percentage",
):
    """
    Draw a heatmap from rows produced by collect_speedup_heatmap_vs_best_csr():
    one row per storage family, one column per speedup threshold, cell =
    percentage (or, with value_kind="count", the raw number) of matrices for
    which the family beats the baseline by more than that threshold.
    """
    if not rows:
        print(f"No data to draw for {filename}, skipping.")
        return
    if value_kind not in ("percentage", "count"):
        raise ValueError(f"Unknown value_kind {value_kind!r}, expected 'percentage' or 'count'")

    _set_latex_rcparams()

    thresholds = list(thresholds)
    labels = [_tex_escape(f"{label} (n={n})") for label, _, _, n in rows]
    value_index = 1 if value_kind == "percentage" else 2
    matrix = np.array([row[value_index] for row in rows])

    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)

    vmax = 100 if value_kind == "percentage" else max(matrix.max(), 1)
    text_threshold = 0.55 * vmax

    fig_height = row_height * len(rows) + 1.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            value = matrix[yi, xi]
            text_color = "white" if value >= text_threshold else INK_PRIMARY
            label_text = f"{value:.0f}\\%" if value_kind == "percentage" else f"{value:.0f}"
            ax.text(
                xi,
                yi,
                label_text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"$>${t:.2f}x" for t in thresholds])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=INK_PRIMARY)
    ax.set_xlabel("Speedup vs. best CSR / cuSPARSE baseline", color=INK_SECONDARY)
    if title:
        ax.set_title(_tex_escape(title), color=INK_PRIMARY, pad=14)

    ax.tick_params(axis="both", length=0, colors=INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if value_kind == "percentage":
        cbar.set_label("Percentage of matrices in the test set", color=INK_SECONDARY)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}\\%"))
    else:
        cbar.set_label("Number of matrices in the test set", color=INK_SECONDARY)
    cbar.outline.set_visible(False)

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# TikZ export for the heatmap (write_speedup_heatmap_tikz). matplotlib's
# imshow()/colormap machinery isn't available in raw TikZ, so cell colors are
# computed here in Python (by interpolating SEQUENTIAL_BLUE, the same palette
# draw_speedup_heatmap() hands to LinearSegmentedColormap) and written out as
# one \definecolor per cell.
# ---------------------------------------------------------------------------


def _hex_to_rgb01(hex_color):
    """"#rrggbb" string -> (r, g, b) tuple with each channel in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _sequential_blue_rgb(t):
    """
    Linearly interpolate the SEQUENTIAL_BLUE palette at t in [0, 1]: t=0 is
    the palette's first (lightest) stop, t=1 its last (darkest) stop, and
    values in between blend the two nearest stops - the TikZ equivalent of
    what LinearSegmentedColormap does for the matplotlib version of the same
    heatmap.
    """
    t = min(max(t, 0.0), 1.0)
    stops = [_hex_to_rgb01(c) for c in SEQUENTIAL_BLUE]
    n = len(stops) - 1  # number of segments between the len(stops) stops
    pos = t * n  # fractional index into `stops`
    i = min(int(pos), n - 1)  # index of the segment's lower stop
    frac = pos - i  # position within that segment, in [0, 1)
    return tuple(stops[i][k] + (stops[i + 1][k] - stops[i][k]) * frac for k in range(3))


def write_speedup_heatmap_tikz(
    rows,
    filename,
    thresholds=DEFAULT_SPEEDUP_THRESHOLDS,
    title=None,
    value_kind="percentage",
    cell_width=1.4,
    cell_height=0.8,
    label_width=4.5,
    standalone=True,
):
    """
    Write the heatmap from collect_speedup_heatmap_vs_best_csr() as a
    self-contained TikZ picture - same cell colors (sampled from
    SEQUENTIAL_BLUE) and text as draw_speedup_heatmap(), but as a plain TikZ
    picture (no pgfplots) so it can be \\input into a paper or poster.

    Layout knobs (cell size, label offsets, legend geometry) are emitted as
    \\providecommand macros at the top of the file and referenced
    symbolically in the coordinates below, so they can be retuned directly
    in the .tex (or overridden from the surrounding document with
    \\renewcommand before the first \\input) without regenerating the file.
    Note \\providecommand only takes effect once per document: if several of
    these heatmaps are \\input together, the first one's macros win unless
    overridden earlier.

    `filename` should end in ".tex" and receives just the tikzpicture, ready
    to \\input. When standalone is True, a second file - same name with a
    "-standalone" suffix - is also written, wrapping the picture in a
    compilable standalone document for quick previewing.
    """
    if not rows:
        print(f"No data to draw for {filename}, skipping.")
        return
    if value_kind not in ("percentage", "count"):
        raise ValueError(f"Unknown value_kind {value_kind!r}, expected 'percentage' or 'count'")

    thresholds = list(thresholds)
    n_rows = len(rows)
    n_cols = len(thresholds)
    # rows entries are (label, percentages, counts, n) - pick the column that
    # matches value_kind.
    value_index = 1 if value_kind == "percentage" else 2
    matrix = [row[value_index] for row in rows]

    # vmax anchors the color scale: percentages always run 0-100, counts run
    # 0-(largest count actually present). text_threshold is the cutoff above
    # which a cell's background is dark enough that white text reads better
    # than the ink-colored default.
    vmax = 100.0 if value_kind == "percentage" else max(max(vals) for vals in matrix)
    vmax = max(vmax, 1)  # guard against vmax=0 (division by zero below)
    text_threshold = 0.55 * vmax

    # Names of the macros defined below, used in the coordinate expressions
    # so the geometry stays symbolic (and thus retunable) in the emitted file.
    m_row_label_gap = "\\SpeedupHeatmapRowLabelGap"
    m_threshold_label_y = "\\SpeedupHeatmapThresholdLabelYOffset"
    m_axis_label_y = "\\SpeedupHeatmapAxisLabelYOffset"
    m_legend_gap = "\\SpeedupHeatmapLegendGap"
    m_legend_width = "\\SpeedupHeatmapLegendWidth"
    m_legend_band_height = "\\SpeedupHeatmapLegendBandHeight"
    m_legend_label_gap = "\\SpeedupHeatmapLegendLabelGap"
    m_title_x_nudge = "\\SpeedupHeatmapTitleXNudge"
    m_title_y_gap = "\\SpeedupHeatmapTitleYGap"

    lines = []
    lines.append(f"\\definecolor{{inkprimary}}{{HTML}}{{{INK_PRIMARY.lstrip('#')}}}")
    lines.append(f"\\providecommand{{\\SpeedupHeatmapCellWidth}}{{{cell_width}cm}}")
    lines.append(f"\\providecommand{{\\SpeedupHeatmapCellHeight}}{{{cell_height}cm}}")
    lines.append(f"\\providecommand{{{m_row_label_gap}}}{{-0.2}}")
    lines.append(f"\\providecommand{{{m_threshold_label_y}}}{{-0.15}}")
    lines.append(f"\\providecommand{{{m_axis_label_y}}}{{-3.5}}")
    lines.append(f"\\providecommand{{{m_legend_gap}}}{{0.6}}")
    lines.append(f"\\providecommand{{{m_legend_width}}}{{0.6}}")
    lines.append(f"\\providecommand{{{m_legend_band_height}}}{{1.0}}")
    lines.append(f"\\providecommand{{{m_legend_label_gap}}}{{0.1}}")
    lines.append(f"\\providecommand{{{m_title_x_nudge}}}{{0.5}}")
    lines.append(f"\\providecommand{{{m_title_y_gap}}}{{0.3}}")
    lines.append(
        "\\begin{tikzpicture}[x=\\SpeedupHeatmapCellWidth, y=\\SpeedupHeatmapCellHeight]"
    )

    for yi, (label, percentages, counts, n) in enumerate(rows):
        values = percentages if value_kind == "percentage" else counts
        # rows[0] drawn at the top, same top-down convention as the overview
        # bar chart (PosterOverviewGraphs.write_speedup_overview_tikz).
        y = n_rows - 1 - yi
        for xi, value in enumerate(values):
            # xi is also the column's index into `thresholds` (both are
            # built by zipping the same list, see collect_speedup_heatmap_
            # vs_best_csr()), so cell (xi, y) really is (thresholds[xi],
            # rows[yi]).
            r, g, b = _sequential_blue_rgb(value / vmax)
            text_color = "white" if value >= text_threshold else "inkprimary"
            value_text = f"{value:.0f}\\%" if value_kind == "percentage" else f"{value:.0f}"
            lines.append(f"  \\definecolor{{cellcolor}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}")
            lines.append(f"  \\fill[cellcolor] ({xi},{y}) rectangle ++(1,1);")
            lines.append(
                f"  \\node[text={text_color}] at ({xi + 0.5},{y + 0.5}) {{{value_text}}};"
            )
        lines.append(
            f"  \\node[anchor=east, text=inkprimary] at ({m_row_label_gap},{y + 0.5}) "
            f"{{{_tex_escape(f'{label} (n={n})')}}};"
        )

    for xi, t in enumerate(thresholds):
        lines.append(
            f"  \\node[anchor=north east, rotate=45, text=inkprimary] "
            f"at ({xi + 0.5},{m_threshold_label_y}) {{$>${t:.2f}x}};"
        )
    lines.append(
        f"  \\node[anchor=north, text=inkprimary] at ({n_cols / 2},{m_axis_label_y}) "
        f"{{Speedup vs. best CSR / cuSPARSE baseline}};"
    )

    # Legend band height is fixed (independent of n_rows) so the value labels
    # always have enough room, regardless of how many families are plotted;
    # bottom-aligned with the grid and increasing upward, like a colorbar.
    legend_steps = 6
    band_height = 1.0  # kept in sync with \SpeedupHeatmapLegendBandHeight's default
    legend_height = legend_steps * band_height
    legend_x = f"{n_cols}+{m_legend_gap}"
    for i in range(legend_steps):
        # Band i covers the value range [v0, v1); it's filled with the color
        # a cell at its midpoint would get, and labeled with its lower bound
        # v0 (the top-most band's upper bound vmax is labeled separately,
        # after the loop, as the top tick).
        v0 = vmax * i / legend_steps
        v1 = vmax * (i + 1) / legend_steps
        r, g, b = _sequential_blue_rgb((v0 + v1) / 2 / vmax)
        y0 = f"{i}*{m_legend_band_height}" if i else "0"
        lines.append(f"  \\definecolor{{legendcolor}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}")
        lines.append(
            f"  \\fill[legendcolor] ({legend_x},{y0}) rectangle "
            f"++({m_legend_width},{m_legend_band_height});"
        )
        label_text = f"{v0:.0f}\\%" if value_kind == "percentage" else f"{v0:.0f}"
        lines.append(
            f"  \\node[anchor=west, text=inkprimary] at "
            f"({legend_x}+{m_legend_width}+{m_legend_label_gap},{y0}) {{{label_text}}};"
        )
    top_label = f"{vmax:.0f}\\%" if value_kind == "percentage" else f"{vmax:.0f}"
    lines.append(
        f"  \\node[anchor=west, text=inkprimary] at "
        f"({legend_x}+{m_legend_width}+{m_legend_label_gap},"
        f"{legend_steps}*{m_legend_band_height}) {{{top_label}}};"
    )

    top_extent = max(n_rows, legend_height)
    if title:
        lines.append(
            f"  \\node[anchor=south, text=inkprimary, font=\\bfseries] "
            f"at ({n_cols / 2}+{m_title_x_nudge},{top_extent}+{m_title_y_gap}) "
            f"{{{_tex_escape(title)}}};"
        )

    lines.append("\\end{tikzpicture}")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    if standalone:
        _write_tikz_standalone_wrapper(filename)


def speedup_heatmap_vs_best_csr(
    df,
    accelerator_devices,
    output_dir="Poster",
    thresholds=DEFAULT_SPEEDUP_THRESHOLDS,
    include_cpu_csr=False,
):
    """
    Entry point: for every accelerator device, draw a heatmap of storage
    families (Ellpack, RowMajor/ColumnMajor SlicedEllpack, BiEllpack,
    ChunkedEllpack) vs. speedup thresholds, showing what fraction of matrices
    beat the best-CSR-or-cuSPARSE baseline by more than each threshold - one
    version with percentages, one with the raw matrix counts. When
    include_cpu_csr is True, the baseline also considers the best plain CSR
    time on the CPU, and the output files get a "-incl-cpu-csr" suffix.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    suffix = "-incl-cpu-csr" if include_cpu_csr else ""
    title_suffix = " (incl. best CSR on CPU)" if include_cpu_csr else ""
    for device in accelerator_devices:
        rows = collect_speedup_heatmap_vs_best_csr(
            df, device, thresholds=thresholds, include_cpu_csr=include_cpu_csr
        )
        if not rows:
            continue
        print(f"Writing poster speedup heatmap for {device}{suffix} ({len(rows)} families)")
        draw_speedup_heatmap(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-heatmap-vs-best-csr{suffix}-{device}.pdf"
            ),
            thresholds=thresholds,
            title=f"TNL formats vs. best CSR / cuSPARSE baseline on {device}{title_suffix}",
            value_kind="percentage",
        )
        draw_speedup_heatmap(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-heatmap-vs-best-csr{suffix}-{device}-counts.pdf"
            ),
            thresholds=thresholds,
            title=f"TNL formats vs. best CSR / cuSPARSE baseline on {device}{title_suffix}",
            value_kind="count",
        )
        write_speedup_heatmap_tikz(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-heatmap-vs-best-csr{suffix}-{device}.tex"
            ),
            thresholds=thresholds,
            title=f"TNL formats vs. best CSR / cuSPARSE baseline on {device}{title_suffix}",
            value_kind="percentage",
        )
        write_speedup_heatmap_tikz(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-heatmap-vs-best-csr{suffix}-{device}-counts.tex"
            ),
            thresholds=thresholds,
            title=f"TNL formats vs. best CSR / cuSPARSE baseline on {device}{title_suffix}",
            value_kind="count",
        )


# ---------------------------------------------------------------------------
# Per-matrix statistics report (Markdown): names the individual matrices
# behind the heatmap's percentages - the matrices where some storage family
# beats the best-CSR/cuSPARSE baseline by at least `threshold`, together with
# each matrix's per-matrix statistics for context.
# ---------------------------------------------------------------------------

# Per-matrix statistics computed by MatrixStatistics.h and emitted via
# spmv.h's setMetadataColumns(), in the exact key spelling used there - see
# write_speedup_matrices(), which lists these (with their dataset-wide
# min/max/avg) next to every matrix in the Markdown report.
MATRIX_STAT_COLUMNS = [
    "nonzeros/row average",
    "nonzeros/row std_dev",
    "nonzeros/row P50",
    "nonzeros/row P95",
    "nonzeros/row P99",
    "nonzeros/row R50",
    "nonzeros/row R95",
    "nonzeros/row R99",
    "warps/row average",
    "rho ELL",
    "rho SELL 2",
    "rho SELL 4",
    "rho SELL 8",
    "rho SELL 16",
    "rho SELL 32",
    "eta SELL 2",
    "eta SELL 4",
    "eta SELL 8",
    "eta SELL 16",
    "eta SELL 32",
    "entropy",
    "Gini",
]


def _matrix_stat_series(df):
    """{stat: numeric Series indexed like df} for whichever of
    MATRIX_STAT_COLUMNS are present as columns in df."""
    series = {}
    for stat in MATRIX_STAT_COLUMNS:
        col = (stat, "", "", "", "")
        if col in df.columns:
            series[stat] = pd.to_numeric(df[col], errors="coerce")
    return series


def _matrix_stat_summary(stat_series):
    """{stat: (min, max, mean)} over all matrices with a valid value for it."""
    summary = {}
    for stat, s in stat_series.items():
        valid = s.dropna()
        if not valid.empty:
            summary[stat] = (valid.min(), valid.max(), valid.mean())
    return summary


def collect_speedup_matrices_vs_best_csr(
    df,
    device,
    family_filters=None,
    csr_filter=_is_csr_baseline_format,
    threshold=1.1,
    include_cpu_csr=False,
):
    """
    For every storage family in family_filters, find the matrices on `device`
    where the family's best time (best launch config, and best format within
    the family - e.g. best slice size for SlicedEllpack) beats the
    best-CSR-or-cuSPARSE baseline (see collect_speedup_heatmap_vs_best_csr(),
    including the include_cpu_csr option) by at least `threshold` (e.g. 1.1
    means "at least 10% faster").

    Returns a dict {label: [(matrix_name, speedup, rows, columns, nonzeros,
    stats), ...]}, each list sorted by descending speedup, skipping families
    with no qualifying matrix. `stats` is {stat_name: value} for whichever of
    MATRIX_STAT_COLUMNS are available for that matrix.
    """
    if family_filters is None:
        family_filters = FAMILY_FILTERS

    baseline_time = _best_csr_cusparse_baseline_time(
        df, device, csr_filter=csr_filter, include_cpu_csr=include_cpu_csr
    )
    if baseline_time is None:
        return {}

    matrix_names = df[("Matrix name", "", "", "", "")]
    rows_col = pd.to_numeric(df[("rows", "", "", "", "")], errors="coerce")
    columns_col = pd.to_numeric(df[("columns", "", "", "", "")], errors="coerce")
    nonzeros_col = pd.to_numeric(df[("nonzeros", "", "", "", "")], errors="coerce")
    stat_series = _matrix_stat_series(df)

    result = {}
    for label, family_ok in family_filters.items():
        family_time = _min_time_per_matrix(df, device, family_ok)
        if family_time is None:
            continue
        speedup = (baseline_time / family_time).dropna()
        qualifying = speedup[speedup >= threshold].sort_values(ascending=False)
        if qualifying.empty:
            continue
        result[label] = [
            (
                matrix_names.loc[idx],
                speedup_value,
                rows_col.loc[idx],
                columns_col.loc[idx],
                nonzeros_col.loc[idx],
                {stat: s.loc[idx] for stat, s in stat_series.items()},
            )
            for idx, speedup_value in qualifying.items()
        ]
    return result


def write_speedup_matrices(
    df,
    accelerator_devices,
    output_dir="Poster",
    threshold=1.1,
    include_cpu_csr=False,
):
    """
    Entry point: for every accelerator device, write a Markdown file naming
    the matrices where some TNL storage family beats the best CSR / cuSPARSE
    baseline by at least `threshold` (default: 1.1x) - formats with the
    highest speedups first, and within each format, matrices sorted by
    descending speedup. When include_cpu_csr is True, the baseline also
    considers the best plain CSR time on the CPU, and the output file gets a
    "-incl-cpu-csr" suffix.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    stat_summary = _matrix_stat_summary(_matrix_stat_series(df))
    suffix = "-incl-cpu-csr" if include_cpu_csr else ""
    baseline_note = " (baseline includes the best CSR time on the CPU)" if include_cpu_csr else ""
    for device in accelerator_devices:
        families = collect_speedup_matrices_vs_best_csr(
            df, device, threshold=threshold, include_cpu_csr=include_cpu_csr
        )
        if not families:
            continue
        # Sort families by their own top speedup (each family's matrix list
        # is already sorted descending by collect_speedup_matrices_vs_best_
        # csr(), so item[1][0] is a family's best-matching matrix and
        # item[1][0][1] its speedup) - "families with the highest speedups
        # first", per the docstring above.
        families = sorted(
            families.items(), key=lambda item: item[1][0][1], reverse=True
        )
        filename = os.path.join(output_dir, f"speedup-matrices{suffix}-{device}.md")
        print(f"Writing poster speedup matrix list for {device}{suffix} to {filename}")
        with open(filename, "w") as f:
            f.write(
                f"# Matrices with speedup >= {threshold:.2f}x vs. best CSR / "
                f"cuSPARSE baseline on {device}{baseline_note}\n\n"
            )
            for label, matrices in families:
                f.write(f"## {label}\n\n")
                for matrix_name, speedup, rows, columns, nonzeros, stats in matrices:
                    f.write(
                        f"- `{matrix_name}` -- {int(rows)} x {int(columns)}, "
                        f"{int(nonzeros)} nonzeros (speedup: {speedup:.2f}x)\n"
                    )
                    for stat in MATRIX_STAT_COLUMNS:
                        value = stats.get(stat)
                        if value is None or pd.isna(value) or stat not in stat_summary:
                            continue
                        stat_min, stat_max, stat_mean = stat_summary[stat]
                        f.write(
                            f"  - {stat}: {value:.4g} "
                            f"(min: {stat_min:.4g}, max: {stat_max:.4g}, avg: {stat_mean:.4g})\n"
                        )
                f.write("\n")
