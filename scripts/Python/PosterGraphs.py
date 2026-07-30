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
  - speedup_overview_variants: same breakdown, but one chart per
    "Binary"/"Symmetric"/"Sorted" variant tag, comparing each variant against
    its own non-variant counterpart (e.g. "Sorted CSR" vs "CSR") rather than
    against cuSPARSE - these tags are left out of speedup_overview_vs_cusparse
    to keep that chart to one row per base format/launch-config.
  - speedup_heatmap_vs_best_csr: for every accelerator device, a heatmap with
    one row per "storage family" (Ellpack, RowMajor/ColumnMajor SlicedEllpack,
    BiEllpack, ChunkedEllpack - each taken at its best launch config and, for
    SlicedEllpack, best slice size) and one column per speedup threshold,
    showing the percentage of matrices where the family beats the best of
    (plain CSR, cuSPARSE) by more than that threshold.
  - cumulative_coverage_vs_best: for every accelerator device, a waterfall
    chart that starts from cuSPARSE alone and adds one storage family at a
    time (Best CSR, then Ellpack, SlicedEllpack, BiEllpack, ChunkedEllpack),
    showing what percentage of matrices are "won" by the portfolio built so
    far - i.e. the portfolio's best time on that matrix equals the true best
    time achievable by any format/library available for that device.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

import LatexLabels

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
    require_substring=None,
    label_fn=None,
):
    """
    Build a list of (label, breakdown) tuples - one per (format, launch_config)
    combination available for the given accelerator device - suitable for
    draw_speedup_overview().

    By default, formats carrying a "Binary"/"Symmetric"/"Sorted"/"Legacy"
    variant tag are left out to keep the overview chart to one row per base
    format/launch-config; pass exclude_substrings=() to include everything.

    If require_substring is given (e.g. "Sorted"), exclude_substrings is
    ignored and only formats containing that substring are kept - use this to
    build a separate overview for one variant tag (see
    speedup_overview_variants_vs_cusparse()).
    """
    if label_fn is None:
        label_fn = default_label
    rows = []
    for format in formats:
        if format in skip_formats:
            continue
        if require_substring is not None:
            if require_substring not in format:
                continue
        elif any(substr in format for substr in exclude_substrings):
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
    subject_label="TNL",
    reference_label="cuSPARSE",
    title=None,
    fig_width=8,
    row_height=0.55,
):
    """
    Draw a 100% stacked horizontal bar chart from rows produced by
    collect_speedup_overview(): one row per format/launch-config, split into
    "subject_label faster" / "within threshold" / "reference_label faster"
    segments.
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


def speedup_overview_variants(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
    variant_classes=("Symmetric", "Binary", "Sorted"),
):
    """
    Entry point: for every accelerator device and every variant tag in
    variant_classes, draw an overview chart restricted to formats carrying
    that tag (e.g. "Sorted CSR"), comparing each one against its own
    non-variant counterpart (e.g. "Sorted CSR" vs "CSR") using the "speed-up"
    vs. "non-binary"/"non-symmetric"/"non-sorted" column that
    tnl-spmv-benchmark-make-tables-json.py already computes for these
    formats.

    These formats are excluded from speedup_overview_vs_cusparse() to keep
    the main overview to one row per base format/launch-config, so without
    this function they would never show up in the poster output at all.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for variant_class in variant_classes:
        reference = f"non-{variant_class.lower()}"
        for device in accelerator_devices:
            rows = collect_speedup_overview(
                df,
                formats,
                launch_configs,
                device,
                reference=reference,
                threshold=threshold,
                require_substring=variant_class,
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


# "Plain" CSR variants (no Binary/Symmetric compression, which change the
# problem/data assumptions) - these form the "best CSR" half of the baseline
# for speedup_heatmap_vs_best_csr(): CSR, Adaptive CSR, Sorted CSR, Sorted
# Adaptive CSR and their launch configs.
CSR_BASELINE_EXCLUDE_TAGS = ("Binary", "Symmetric", "Legacy", "Best")

# One row per storage family in the heatmap. RowMajor/ColumnMajor SlicedEllpack
# fold their slice-size suffix (e.g. "RowMajor SlicedEllpack 16") into one row
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


def collect_speedup_heatmap_vs_best_csr(
    df,
    device,
    family_filters=None,
    csr_filter=_is_csr_baseline_format,
    thresholds=range(0, 101, 10),
):
    """
    Build heatmap rows for one device: for every storage family in
    family_filters, the percentage of matrices for which the family's best
    time (best launch config, and best format within the family - e.g. best
    slice size for SlicedEllpack) beats the baseline by more than each
    threshold in `thresholds`.

    The baseline for a matrix is whichever is faster: the best "plain" CSR
    variant (see CSR_BASELINE_EXCLUDE_TAGS) or cuSPARSE - i.e. the best
    reference a TNL user could actually reach for on that matrix.

    Returns a list of (label, percentages, counts, n) tuples, skipping
    families with no data for this device. `counts` are the raw number of
    matrices behind each `percentages` entry (percentages[i] == 100 *
    counts[i] / n); draw_speedup_heatmap() can display either.
    """
    if family_filters is None:
        family_filters = FAMILY_FILTERS

    csr_time = _min_time_per_matrix(df, device, csr_filter)
    cusparse_time = _min_time_per_matrix(df, device, lambda format: format == "cusparse")
    if csr_time is None and cusparse_time is None:
        return []
    if csr_time is None:
        baseline_time = cusparse_time
    elif cusparse_time is None:
        baseline_time = csr_time
    else:
        baseline_time = pd.concat([csr_time, cusparse_time], axis=1).min(axis=1)

    thresholds = list(thresholds)
    rows = []
    for label, family_ok in family_filters.items():
        family_time = _min_time_per_matrix(df, device, family_ok)
        if family_time is None:
            continue
        speedup = baseline_time / family_time
        valid = speedup.notna()
        total = int(valid.sum())
        if total == 0:
            continue
        counts = [int((speedup[valid] > 1 + t / 100.0).sum()) for t in thresholds]
        percentages = [100.0 * c / total for c in counts]
        rows.append((label, percentages, counts, total))
    return rows


def draw_speedup_heatmap(
    rows,
    filename,
    thresholds=range(0, 101, 10),
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
    ax.set_xticklabels([f"$>${t}\\%" for t in thresholds])
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
        filters_so_far = tuple(portfolio_filters)
        portfolio_ok = lambda format, fns=filters_so_far: any(fn(format) for fn in fns)
        portfolio_time = _min_time_per_matrix(df, device, portfolio_ok)
        if portfolio_time is None:
            portfolio_values = np.full(len(df), np.inf)
        else:
            portfolio_values = portfolio_time.fillna(np.inf).to_numpy()
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


def speedup_heatmap_vs_best_csr(
    df,
    accelerator_devices,
    output_dir="Poster",
    thresholds=range(0, 101, 10),
):
    """
    Entry point: for every accelerator device, draw a heatmap of storage
    families (Ellpack, RowMajor/ColumnMajor SlicedEllpack, BiEllpack,
    ChunkedEllpack) vs. speedup thresholds, showing what fraction of matrices
    beat the best-CSR-or-cuSPARSE baseline by more than each threshold - one
    version with percentages, one with the raw matrix counts.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_speedup_heatmap_vs_best_csr(df, device, thresholds=thresholds)
        if not rows:
            continue
        print(f"Writing poster speedup heatmap for {device} ({len(rows)} families)")
        draw_speedup_heatmap(
            rows,
            filename=os.path.join(output_dir, f"speedup-heatmap-vs-best-csr-{device}.pdf"),
            thresholds=thresholds,
            title=f"TNL formats vs. best CSR / cuSPARSE baseline on {device}",
            value_kind="percentage",
        )
        draw_speedup_heatmap(
            rows,
            filename=os.path.join(
                output_dir, f"speedup-heatmap-vs-best-csr-{device}-counts.pdf"
            ),
            thresholds=thresholds,
            title=f"TNL formats vs. best CSR / cuSPARSE baseline on {device}",
            value_kind="count",
        )
