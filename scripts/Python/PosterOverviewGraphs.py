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

Entry points:
  - speedup_overview_vs_cusparse: for every TNL format/launch-config on a
    given accelerator device, computes the percentage of matrices where the
    format is clearly faster than cuSPARSE, comparable (within a threshold),
    or clearly slower - and draws it as a 100% stacked horizontal bar chart.
    Alongside the PDF, also writes the same chart as a self-contained TikZ
    picture (write_speedup_overview_tikz) - a ".tex" file ready to \\input
    into a paper/poster (bar width/row height, and other layout offsets, are
    \\providecommand macros so they're retunable without regenerating the
    file), plus a "-standalone.tex" wrapper for quick previewing.
  - speedup_overview_csr_vs_cusparse: the same chart (PDF + TikZ), restricted
    to CSR-family formats only - a narrower companion to
    speedup_overview_vs_cusparse for when only the CSR kernels matter.
  - speedup_overview_variants: same breakdown, but one chart per
    "Binary"/"Symmetric"/"Sorted" variant tag, comparing each variant against
    its own non-variant counterpart (e.g. "Sorted CSR" vs "CSR") rather than
    against cuSPARSE - these tags are left out of speedup_overview_vs_cusparse
    to keep that chart to one row per base format/launch-config.
  - speedup_overview_csr_binary_vs_nonbinary: the Binary-vs-non-binary slice
    of speedup_overview_variants (PDF + TikZ), restricted to CSR-family
    formats only (i.e. "Binary CSR" vs "CSR") - the Binary/CSR-only
    counterpart to speedup_overview_csr_vs_cusparse.
  - speedup_overview_sorted_segments: the impact of sorted segments (PDF +
    TikZ), one row per class (CSR, RowMajor/ColumnMajor SlicedEllpack)
    rather than one row per format/launch-config - each row compares the
    best time achievable with sorted segments against the best time
    achievable without, within that same class (see
    collect_sorted_segments_overview() / SORTED_SEGMENTS_CLASSES).
"""

import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

import LatexLabels
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
)

# ---------------------------------------------------------------------------
# Speed-up breakdown: classifying a series of speed-up values into
# "faster" / "similar" / "slower" percentages. Every chart in this module
# (overview bar charts, sorted-segments chart) reduces to this one formula,
# so it lives in a single place - check the math here once, trust it
# everywhere else.
# ---------------------------------------------------------------------------


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
    no valid values at all.
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
    }


def compute_speedup_breakdown(df, format, device, launch_config, reference="cusparse", threshold=0.10):
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
    reference="cusparse",
    threshold=0.10,
    skip_formats=("cusparse", "CSR Best"),
    exclude_substrings=("Binary", "Symmetric", "Sorted", "Legacy"),
    require_substring=None,
    format_filter=None,
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

    format_filter, if given, is an extra predicate applied on top of the
    above (e.g. restrict to CSR-family formats only, see
    speedup_overview_csr_vs_cusparse()).
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
        if format_filter is not None and not format_filter(format):
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
    reference_label="cuSPARSE",
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
    Alongside the PDF, also writes the same chart as a self-contained TikZ
    picture (write_speedup_overview_tikz) - a ".tex" file ready to \\input
    into a paper/poster, plus a "-standalone.tex" wrapper for previewing.
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
        write_speedup_overview_tikz(
            rows,
            filename=os.path.join(output_dir, f"speedup-vs-cusparse-{device}.tex"),
            threshold=threshold,
            reference_label="cuSPARSE",
            title=f"TNL vs. cuSPARSE on {device}",
        )


def speedup_overview_csr_vs_cusparse(
    df,
    formats,
    launch_configs,
    accelerator_devices,
    output_dir="Poster",
    threshold=0.10,
):
    """
    Entry point: for every accelerator device, draw a narrower companion to
    speedup_overview_vs_cusparse() restricted to CSR-family formats only
    (still excluding Binary/Symmetric/Sorted/Legacy variants, same as the
    main overview) - PDF plus the same TikZ output.
    """
    if "cusparse" not in formats:
        print("No cusparse results found, skipping poster CSR speedup overview.")
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for device in accelerator_devices:
        rows = collect_speedup_overview(
            df,
            formats,
            launch_configs,
            device,
            reference="cusparse",
            threshold=threshold,
            format_filter=_is_plain_csr_format,
        )
        if not rows:
            continue
        print(f"Writing poster CSR speedup overview for {device} ({len(rows)} configurations)")
        draw_speedup_overview(
            rows,
            filename=os.path.join(output_dir, f"speedup-csr-vs-cusparse-{device}.pdf"),
            threshold=threshold,
            subject_label="CSR",
            reference_label="cuSPARSE",
            title=f"CSR vs. cuSPARSE on {device}",
        )
        write_speedup_overview_tikz(
            rows,
            filename=os.path.join(output_dir, f"speedup-csr-vs-cusparse-{device}.tex"),
            threshold=threshold,
            subject_label="CSR",
            reference_label="cuSPARSE",
            title=f"CSR vs. cuSPARSE on {device}",
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
    are excluded, unlike speedup_overview_csr_vs_cusparse() which only needs
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
