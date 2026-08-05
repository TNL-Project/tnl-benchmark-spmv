"""
Poster/paper chart: the cumulative-coverage waterfall - "if a user could pick
from cuSPARSE alone, how many matrices get the true best time? What if we
also let them pick Best CSR? Also Ellpack? ..." See PosterGraphsCommon.py for
the shared color palette and baseline-filter building blocks,
PosterOverviewGraphs.py for the row/bar overview charts,
PosterHeatmapGraphs.py for the threshold heatmap.

This module is a scratch space: functions here are added/removed/reshaped
while we figure out how the results should be presented. Nothing here is
expected to be stable API - once we settle on a presentation, it should move
into Report.py as a proper, systematic report.

Entry point:
  - cumulative_coverage_vs_best: for every accelerator device, a waterfall
    chart that starts from cuSPARSE alone and adds one storage family at a
    time (Best CSR, then Ellpack, SlicedEllpack, BiEllpack, ChunkedEllpack),
    showing what percentage of matrices are "won" by the portfolio built so
    far - i.e. the portfolio's best time on that matrix equals the true best
    time achievable by any format/library available for that device.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from PosterGraphsCommon import (
    COLOR_FASTER,
    SEQUENTIAL_BLUE,
    INK_PRIMARY,
    INK_SECONDARY,
    INK_MUTED,
    GRIDLINE,
    _tex_escape,
    _set_latex_rcparams,
    _min_time_per_matrix,
    FAMILY_FILTERS,
    _is_csr_baseline_format,
)


def _is_sliced_ellpack_format(format):
    return format.startswith("RowMajor SlicedEllpack") or format.startswith(
        "ColumnMajor SlicedEllpack"
    )


# Cumulative portfolio for cumulative_coverage_vs_best(): each step's filter
# is ADDED to every filter before it (see collect_cumulative_coverage()), so
# this reads as "start from cuSPARSE alone, then also allow Best CSR, then
# also allow Ellpack, ...". The last label names the fully-grown portfolio
# rather than just the last format added, since that is the number that
# actually matters on the poster.
DEFAULT_COVERAGE_STEPS = (
    ("cuSPARSE", lambda format: format == "cusparse"),
    ("+ Best CSR", _is_csr_baseline_format),
    ("+ Ellpack", FAMILY_FILTERS["Ellpack"]),
    ("+ SlicedEllpack", _is_sliced_ellpack_format),
    ("+ BiEllpack", FAMILY_FILTERS["BiEllpack"]),
    ("Best of all TNL formats", FAMILY_FILTERS["ChunkedEllpack"]),
)


def collect_cumulative_coverage(df, device, steps=None):
    """
    Build waterfall rows for one device: starting from cuSPARSE alone and
    growing the portfolio one step at a time (see DEFAULT_COVERAGE_STEPS),
    compute for each step the percentage of matrices "won" by the portfolio
    built so far - i.e. the portfolio's best time on that matrix equals the
    true best time achievable by ANY format/library available for this
    device (not just the ones in `steps`), so 100% is only reached if the
    portfolio's formats happen to cover every actual winner.

    Returns a list of (label, cumulative_pct, delta_pct, n) tuples, or [] if
    there is no data at all for this device.
    """
    if steps is None:
        steps = DEFAULT_COVERAGE_STEPS

    global_time = _min_time_per_matrix(df, device, lambda _format: True)
    if global_time is None:
        return []
    valid_global = global_time.notna().to_numpy()
    total = int(valid_global.sum())
    if total == 0:
        return []
    global_values = global_time.to_numpy()

    rows = []
    portfolio_filters = []
    previous_pct = 0.0
    for label, filter_fn in steps:
        portfolio_filters.append(filter_fn)
        # fns=filters_so_far binds *this iteration's* snapshot of the filter
        # list as the lambda's default argument - without it, every
        # portfolio_ok created in this loop would share the same
        # portfolio_filters list object and all end up checking the FINAL,
        # fully-grown set of filters (a classic late-binding-closure bug).
        filters_so_far = tuple(portfolio_filters)
        portfolio_ok = lambda format, fns=filters_so_far: any(fn(format) for fn in fns)
        portfolio_time = _min_time_per_matrix(df, device, portfolio_ok)
        if portfolio_time is None:
            portfolio_values = np.full(len(df), np.inf)
        else:
            # Missing/NaN times (format not run for this matrix) become +inf
            # so they can never "win" the comparison below.
            portfolio_values = portfolio_time.fillna(np.inf).to_numpy()
        # A matrix counts as "won" if the portfolio's best time here is the
        # same value as the best time achievable by ANY format/library for
        # this device (global_time) - i.e. the portfolio already contains
        # this matrix's true winner. np.isclose rather than == to tolerate
        # floating-point noise from the min() reductions on both sides.
        won = np.isclose(portfolio_values, global_values, rtol=1e-9, atol=0) & valid_global
        cumulative_pct = 100.0 * int(won.sum()) / total
        rows.append((label, cumulative_pct, cumulative_pct - previous_pct, total))
        previous_pct = cumulative_pct
    return rows


def draw_cumulative_coverage(rows, filename, title=None, fig_width=8, fig_height=4.5):
    """
    Draw a waterfall chart from rows produced by collect_cumulative_coverage():
    the first and last bars stand on their own (cuSPARSE alone, then the
    fully-grown portfolio); the bars in between float between the running
    totals and are labeled with the coverage gained by that one step.
    """
    if not rows:
        print(f"No data to draw for {filename}, skipping.")
        return

    _set_latex_rcparams()

    labels = [_tex_escape(label) for label, _, _, _ in rows]
    cumulative = [cum for _, cum, _, _ in rows]
    deltas = [delta for _, _, delta, _ in rows]
    n = rows[0][3]
    last = len(rows) - 1

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    for i in range(len(rows)):
        if i == 0 or i == last:
            bottom, height = 0.0, cumulative[i]
            color = SEQUENTIAL_BLUE[3] if i == 0 else COLOR_FASTER
            label_text = f"{cumulative[i]:.0f}\\%"
        else:
            bottom, height = cumulative[i - 1], deltas[i]
            color = SEQUENTIAL_BLUE[7]
            label_text = f"+{deltas[i]:.0f}\\%"
        ax.bar(i, height, bottom=bottom, width=0.6, color=color, edgecolor="white", linewidth=1.2)
        ax.text(i, bottom + height + 1.5, label_text, ha="center", va="bottom", fontsize=10, color=INK_PRIMARY)

    for i in range(len(rows) - 1):
        ax.plot(
            [i + 0.3, i + 1 - 0.3],
            [cumulative[i], cumulative[i]],
            color=INK_MUTED,
            linewidth=1.0,
            linestyle="--",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=INK_PRIMARY)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}\\%"))
    ax.set_ylabel(f"Matrices where the portfolio has the overall best time (n={n})", color=INK_SECONDARY)
    if title:
        ax.set_title(_tex_escape(title), color=INK_PRIMARY, pad=14)

    ax.tick_params(axis="both", length=0, colors=INK_PRIMARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def cumulative_coverage_vs_best(df, accelerator_devices, output_dir="Poster", steps=None):
    """
    Entry point: for every accelerator device, draw the cuSPARSE-to-"all TNL
    formats" waterfall chart (see module docstring / DEFAULT_COVERAGE_STEPS).
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_cumulative_coverage(df, device, steps=steps)
        if not rows:
            continue
        print(f"Writing poster cumulative coverage for {device} ({len(rows)} steps)")
        draw_cumulative_coverage(
            rows,
            filename=os.path.join(output_dir, f"cumulative-coverage-{device}.pdf"),
            title=f"Cumulative coverage of the overall best time on {device}",
        )
