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
  - cumulative_coverage_vs_best: for every accelerator device, a waterfall
    chart that starts from cuSPARSE alone and adds one storage family at a
    time (Best CSR, then Ellpack, SlicedEllpack, BiEllpack, ChunkedEllpack),
    showing what percentage of matrices are "won" by the portfolio built so
    far - i.e. the portfolio's best time on that matrix equals the true best
    time achievable by any format/library available for that device.
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

import itertools
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
        y = n_rows - 1 - yi
        left = 0.0
        for value, colorname, text_color in (
            (breakdown["faster"], "colorfaster", "white"),
            (breakdown["similar"], "colorsimilar", "inkprimary"),
            (breakdown["slower"], "colorslower", "white"),
        ):
            width = value / 100.0
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
        base, ext = os.path.splitext(filename)
        standalone_filename = f"{base}-standalone{ext}"
        with open(standalone_filename, "w") as f:
            f.write("\\documentclass{standalone}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage{tikz}\n")
            f.write("\\begin{document}\n")
            f.write(f"\\input{{{os.path.basename(filename)}}}\n")
            f.write("\\end{document}\n")


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
        speedup = (nonsorted_time / sorted_time).dropna()
        total = len(speedup)
        if total == 0:
            continue
        faster = int((speedup > 1 + threshold).sum())
        slower = int((speedup < 1 - threshold).sum())
        similar = total - faster - slower
        breakdown = {
            "faster": 100.0 * faster / total,
            "similar": 100.0 * similar / total,
            "slower": 100.0 * slower / total,
            "n": total,
        }
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


# Speedup axis for the heatmap: 1.00x, 1.05x, ..., 1.50x.
DEFAULT_SPEEDUP_THRESHOLDS = [round(1.0 + 0.05 * i, 2) for i in range(11)]


def _best_csr_cusparse_baseline_time(
    df, device, csr_filter=_is_csr_baseline_format, include_cpu_csr=False
):
    """
    Per-matrix best (lowest) time among: the best "plain" CSR variant on
    `device` (see CSR_BASELINE_EXCLUDE_TAGS), cuSPARSE on `device`, and - when
    include_cpu_csr is True - also the best plain CSR variant on the CPU (i.e.
    a TNL user willing to fall back to a CPU run would take that if it beats
    everything available on `device`). Returns None if none of these are
    available for this device.
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
    variant on `device` (see CSR_BASELINE_EXCLUDE_TAGS), cuSPARSE on `device`,
    and - when include_cpu_csr is True - also the best plain CSR variant on
    the CPU - i.e. the best reference a TNL user could actually reach for on
    that matrix.

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


def _hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _sequential_blue_rgb(t):
    """Linearly interpolate the SEQUENTIAL_BLUE palette at t in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    stops = [_hex_to_rgb01(c) for c in SEQUENTIAL_BLUE]
    n = len(stops) - 1
    pos = t * n
    i = min(int(pos), n - 1)
    frac = pos - i
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
    value_index = 1 if value_kind == "percentage" else 2
    matrix = [row[value_index] for row in rows]

    vmax = 100.0 if value_kind == "percentage" else max(max(vals) for vals in matrix)
    vmax = max(vmax, 1)
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
        y = n_rows - 1 - yi
        for xi, value in enumerate(values):
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
        base, ext = os.path.splitext(filename)
        standalone_filename = f"{base}-standalone{ext}"
        with open(standalone_filename, "w") as f:
            f.write("\\documentclass{standalone}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage{tikz}\n")
            f.write("\\begin{document}\n")
            f.write(f"\\input{{{os.path.basename(filename)}}}\n")
            f.write("\\end{document}\n")


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
