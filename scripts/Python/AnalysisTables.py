"""
Tables analyzing the results, written with --with-tables to the directory
Tables. Each table is written as

  - <name>.html: a page of its own,
  - <name>.tex: just the tabular environment (it needs the booktabs package),
    ready to \\input into a paper,
  - <name>-standalone.tex: a document with the table for a preview,
  - <name>.csv: the numbers the table is made of.

The tables for the transposed matrices have the suffix "-transposed".

Tables:

  - winners-by-size: for each GPU and each class of matrices by the number of
    nonzeros, the percentage of matrices on which each implementation (TNL
    CSR, TNL ELLPACK, the vendor library, Ginkgo, Hypre) is the fastest one,
    see winners_by_size_table().
"""

import html
import os
import re
import numpy as np
import pandas as pd

from VendorLibraries import vendor_label
from PosterGraphsCommon import (
    _min_time_per_matrix,
    _is_csr_baseline_format,
    _vendor_format_filter,
    FAMILY_FILTERS,
)

MATRIX_NAME = ("Matrix name", "", "", "", "")
NONZEROS = ("nonzeros", "", "", "", "")

TABLES_DIR = "Tables"


# ---------------------------------------------------------------------------
# Writing a table as HTML and LaTeX
# ---------------------------------------------------------------------------


class Text:
    """
    A text of a cell with different HTML and LaTeX forms. A plain string in a
    cell is used for both and escaped as needed.
    """

    def __init__(self, html_text, latex_text):
        self.html = html_text
        self.latex = latex_text


def _html(cell):
    return cell.html if isinstance(cell, Text) else html.escape(str(cell))


def _latex(cell):
    if isinstance(cell, Text):
        return cell.latex
    return re.sub(r"([%&_#])", r"\\\1", str(cell))


DASH = Text("&mdash;", "--")


class Table:
    """
    A table with a header and groups of rows: each group has a title spanning
    the table (may be None) and rows of cells. The first `left_columns`
    columns are aligned to the left, the others to the right. The cells listed
    in `bold` as (group index, row index, column index) are emphasized.
    """

    def __init__(self, title, header, left_columns=1, note=None):
        self.title = title
        self.header = header
        self.left_columns = left_columns
        self.note = note
        self.groups = []
        self.bold = set()

    def add_group(self, title):
        self.groups.append((title, []))

    def add_row(self, cells, bold_columns=()):
        rows = self.groups[-1][1]
        for column in bold_columns:
            self.bold.add((len(self.groups) - 1, len(rows), column))
        rows.append(cells)

    def write_html(self, file_name):
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            f'<head><meta charset="utf-8"><title>{html.escape(self.title)}</title>',
            "<style>",
            "body { font-family: sans-serif; margin: 2em; }",
            "table { border-collapse: collapse; }",
            "th, td { padding: 0.3em 0.9em; text-align: right; }",
            "th.left, td.left { text-align: left; }",
            "thead th { border-top: 2px solid black; border-bottom: 1px solid black; }",
            "tr.group th { text-align: left; border-top: 1px solid black; }",
            "tbody:last-child tr:last-child td { border-bottom: 2px solid black; }",
            "p.note { max-width: 50em; color: #444; }",
            "</style></head>",
            "<body>",
            f"<h2>{html.escape(self.title)}</h2>",
            "<table>",
            "<thead><tr>",
        ]
        for j, cell in enumerate(self.header):
            css = ' class="left"' if j < self.left_columns else ""
            lines.append(f"  <th{css}>{_html(cell)}</th>")
        lines.append("</tr></thead>")
        for g, (group_title, rows) in enumerate(self.groups):
            lines.append("<tbody>")
            if group_title is not None:
                lines.append(
                    f'<tr class="group"><th colspan="{len(self.header)}">{_html(group_title)}</th></tr>'
                )
            for i, cells in enumerate(rows):
                lines.append("<tr>")
                for j, cell in enumerate(cells):
                    css = ' class="left"' if j < self.left_columns else ""
                    text = _html(cell)
                    if (g, i, j) in self.bold:
                        text = f"<b>{text}</b>"
                    lines.append(f"  <td{css}>{text}</td>")
                lines.append("</tr>")
            lines.append("</tbody>")
        lines.append("</table>")
        if self.note:
            lines.append(f'<p class="note">{html.escape(self.note)}</p>')
        lines += ["</body>", "</html>"]
        with open(file_name, "w") as file:
            file.write("\n".join(lines) + "\n")

    def write_latex(self, file_name):
        """
        Write the tabular environment to file_name and a document including it
        to the file with the suffix "-standalone"
        """
        columns = "l" * self.left_columns + "r" * (len(self.header) - self.left_columns)
        lines = [
            f"% {self.title}",
            "% requires \\usepackage{booktabs}",
            f"\\begin{{tabular}}{{{columns}}}",
            "\\toprule",
            " & ".join(_latex(cell) for cell in self.header) + " \\\\",
        ]
        for g, (group_title, rows) in enumerate(self.groups):
            lines.append("\\midrule")
            if group_title is not None:
                lines.append(
                    f"\\multicolumn{{{len(self.header)}}}{{l}}{{\\textbf{{{_latex(group_title)}}}}} \\\\"
                )
            for i, cells in enumerate(rows):
                texts = []
                for j, cell in enumerate(cells):
                    text = _latex(cell)
                    texts.append(f"\\textbf{{{text}}}" if (g, i, j) in self.bold else text)
                lines.append(" & ".join(texts) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(file_name, "w") as file:
            file.write("\n".join(lines) + "\n")

        base, extension = os.path.splitext(file_name)
        with open(f"{base}-standalone{extension}", "w") as file:
            file.write("\\documentclass{article}\n")
            file.write("\\usepackage[utf8]{inputenc}\n")
            file.write("\\usepackage{booktabs}\n")
            file.write("\\usepackage[margin=1.5cm]{geometry}\n")
            file.write("\\pagestyle{empty}\n")
            file.write("\\begin{document}\n")
            file.write("\\begin{table}\n\\centering\n")
            file.write(f"\\input{{{os.path.basename(file_name)}}}\n")
            file.write(f"\\caption{{{_latex(self.title)}}}\n")
            file.write("\\end{table}\n")
            file.write("\\end{document}\n")

    def write(self, output_dir, name):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.write_html(os.path.join(output_dir, f"{name}.html"))
        self.write_latex(os.path.join(output_dir, f"{name}.tex"))


# ---------------------------------------------------------------------------
# Winners by the size of the matrix
# ---------------------------------------------------------------------------

# (HTML label, LaTeX label, lower bound, upper bound) of the number of nonzeros
NNZ_CLASSES = [
    (Text("&lt;10<sup>5</sup>", "$<10^5$"), 0, 1e5),
    (Text("10<sup>5</sup>&ndash;10<sup>6</sup>", "$10^5$--$10^6$"), 1e5, 1e6),
    (Text("10<sup>6</sup>&ndash;10<sup>7</sup>", "$10^6$--$10^7$"), 1e6, 1e7),
    (Text("&ge;10<sup>7</sup>", "$\\geq 10^7$"), 1e7, np.inf),
]

SORTED_PREFIX = re.compile(r"^(Sigma-\d+-)?Sorted ")


def _is_tnl_csr_format(format):
    """
    CSR and Adaptive CSR of TNL, also with sorted rows, without the binary
    and symmetric variants
    """
    return _is_csr_baseline_format(format)


def _is_tnl_ellpack_format(format):
    """
    Ellpack, SlicedEllpack, BiEllpack and ChunkedEllpack of TNL, also with
    sorted rows, without the binary and symmetric variants
    """
    base = SORTED_PREFIX.sub("", format)
    return any(format_ok(base) for format_ok in FAMILY_FILTERS.values())


# the implementations of the other libraries with one configuration
OTHER_LIBRARIES = [("Ginkgo", "Ginkgo"), ("hypre", "Hypre")]
TNL_IMPLEMENTATIONS = ["TNL CSR", "TNL ELLPACK"]


def collect_implementation_times(df, device):
    """
    Return a dataframe indexed by the matrix name with the best time of each
    implementation on the device over all its formats and launch
    configurations. The implementations without any result on the device are
    left out.
    """
    filters = {"TNL CSR": _is_tnl_csr_format, "TNL ELLPACK": _is_tnl_ellpack_format}
    vendor_ok = _vendor_format_filter(device)
    if vendor_ok is not None:
        filters["Vendor"] = vendor_ok
    for label, format in OTHER_LIBRARIES:
        filters[label] = lambda name, format=format: name == format

    times = pd.DataFrame(index=df[MATRIX_NAME].values)
    for label, format_ok in filters.items():
        best = _min_time_per_matrix(df, device, format_ok)
        if best is not None and best.notna().any():
            times[label] = best.values
    return times


def winners_by_size_table(df, accelerator_devices, transposed=False, output_dir=TABLES_DIR):
    """
    Write the table of the percentages of matrices on which each
    implementation is the fastest one, for each GPU and each class of matrices
    by the number of nonzeros (see NNZ_CLASSES) and for all matrices.

    The time of an implementation on a matrix is its best time over all its
    formats and launch configurations (see collect_implementation_times()).
    Only the matrices with the results of all implementations on all devices
    are counted, so the numbers of matrices are the same for all devices. The
    largest percentage in each row is emphasized, an implementation which is
    not benchmarked on the device has a dash. "TNL total" is the sum of TNL
    CSR and TNL ELLPACK, it is not emphasized.
    """
    times_by_device = {device: collect_implementation_times(df, device) for device in accelerator_devices}
    times_by_device = {device: times for device, times in times_by_device.items() if len(times.columns) > 1}
    if not times_by_device:
        print("No GPU results of several implementations, skipping the table of the winners by size.")
        return
    common = None
    for times in times_by_device.values():
        complete = times.dropna().index
        common = complete if common is None else common.intersection(complete)
    if common.empty:
        print("No matrices with the results of all implementations on all devices, skipping the table.")
        return
    nonzeros = pd.Series(pd.to_numeric(df[NONZEROS], errors="coerce").values, index=df[MATRIX_NAME].values)
    nonzeros = nonzeros.loc[common]

    suffix = "-transposed" if transposed else ""
    matrices = "transposed matrices" if transposed else "original matrices"
    columns = TNL_IMPLEMENTATIONS + ["TNL total", "Vendor"] + [label for label, _ in OTHER_LIBRARIES]
    table = Table(
        f"Percentage of matrices on which each implementation is the fastest one, {matrices}",
        ["Matrix size (nnz)", "N"] + columns,
        note=(
            f"Only the {len(common)} matrices with the results of all implementations on all devices are "
            "counted. The time of an implementation on a matrix is its best time over all its formats and "
            "launch configurations: TNL CSR over CSR and Adaptive CSR, TNL ELLPACK over Ellpack, "
            "SlicedEllpack, BiEllpack and ChunkedEllpack (both also with sorted rows), the vendor library "
            "over all its formats and algorithms, Ginkgo and hypre have one configuration. TNL total is the "
            "sum of TNL CSR and TNL ELLPACK."
        ),
    )
    classes = NNZ_CLASSES + [("All", 0, np.inf)]
    records = []
    ties = 0
    for device, times in times_by_device.items():
        times = times.loc[common]
        winner = times.idxmin(axis=1)
        ties += int((times.eq(times.min(axis=1), axis=0).sum(axis=1) > 1).sum())
        vendor = vendor_label(device)
        table.add_group(f"{device} (vendor: {vendor})" if vendor else device)
        for label, lower, upper in classes:
            in_class = (nonzeros >= lower) & (nonzeros < upper)
            total = int(in_class.sum())
            wins = winner[in_class].value_counts()
            shares = {
                implementation: 100.0 * wins.get(implementation, 0) / total if total else np.nan
                for implementation in times.columns
            }
            shares["TNL total"] = sum(shares[implementation] for implementation in TNL_IMPLEMENTATIONS)
            cells = [label, total]
            bold = []
            best = max(shares[implementation] for implementation in times.columns) if total else None
            for j, column in enumerate(columns):
                if column not in shares or not total:
                    cells.append(DASH)
                    continue
                cells.append(Text(f"{shares[column]:.1f}%", f"{shares[column]:.1f}\\,\\%"))
                if column != "TNL total" and shares[column] == best:
                    bold.append(len(cells) - 1)
            table.add_row(cells, bold)
            class_name = label if isinstance(label, str) else f"[{lower:g}, {upper:g})"
            for implementation in times.columns:
                records.append(
                    {
                        "device": device,
                        "nonzeros": class_name,
                        "matrices": total,
                        "implementation": implementation,
                        "wins": int(wins.get(implementation, 0)),
                        "share": shares[implementation],
                    }
                )

    name = f"winners-by-size{suffix}"
    table.write(output_dir, name)
    pd.DataFrame(records).to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    print(
        f"Table of the winners by the size of the matrix for {len(common)} {matrices} written to "
        f"{os.path.join(output_dir, name)}.html/.tex/.csv"
    )
    if ties:
        print(f"   {ties} ties of the best time, the first implementation in the order of the columns wins.")


def write_tables(df, accelerator_devices, transposed=False, output_dir=TABLES_DIR):
    """
    Entry point: write all tables
    """
    winners_by_size_table(df, accelerator_devices, transposed, output_dir)
