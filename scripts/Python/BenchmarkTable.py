"""
Binary (pickle) storage of the complete benchmark tables for the original and
the transposed matrices, including all computed speed-ups, together with the
metadata needed to analyze them, their export to HTML or Excel and their
flat export to Parquet or CSV.

Written by tnl-spmv-benchmark-parse-logs.py and read by
tnl-spmv-benchmark-analysis-tables.py and the scripts in poster-sc2026/.
"""

import os
import pickle
import html

import pandas as pd

# Table names in the order they are stored and exported
table_names = ["original", "transposed"]

# Maximal number of columns of an Excel sheet
excel_max_columns = 16384


def save_tables(file_name, tables):
    """
    Store the complete tables for the original and the transposed matrices
    in a binary file, so that they can be loaded by load_table(). `tables`
    maps the table name (see table_names) to a dict with the keys "df",
    "formats", "launch_configs" and "accelerator_devices", or to None if
    there are no results for these matrices.
    """
    with open(file_name, "wb") as f:
        pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Complete tables written to {file_name}")


def load_table(file_name, transposed=False):
    """
    Load the complete table for the original or the transposed matrices and
    its metadata stored by save_tables()
    """
    name = "transposed" if transposed else "original"
    print(f"Loading complete table for the {name} matrices from {file_name} ...")
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    # files written before the transposed matrices were supported hold just one table
    if "df" in data:
        if transposed:
            raise ValueError(f"{file_name} does not contain a table for the transposed matrices.")
        table = data
    else:
        table = data.get(name)
        if table is None:
            raise ValueError(f"{file_name} does not contain a table for the {name} matrices.")
    return table["df"], table["formats"], table["launch_configs"], table["accelerator_devices"]


def get_export_df(df):
    """
    Prepare the table for the export: rows are indexed by the matrix name and
    the columns without any value (e.g. a format which was not run on some
    device) are left out
    """
    name_column = ("Matrix name", "", "", "", "")
    df = df.dropna(axis="columns", how="all")
    df = df.set_index(name_column)
    df.index.name = "Matrix name"
    return df


def export_tables(file_name, tables):
    """
    Export the complete tables to an HTML file (one table after another) or to
    an Excel file (one sheet per table), depending on the file extension
    """
    extension = os.path.splitext(file_name)[1].lower()
    exported = {name: get_export_df(tables[name]["df"]) for name in table_names if tables.get(name)}
    print(f"Exporting the complete tables to {file_name} ...")
    if extension in [".html", ".htm"]:
        with open(file_name, "w") as f:
            f.write("<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"><title>SpMV benchmark</title></head>\n<body>\n")
            for name, df in exported.items():
                f.write(f"<h2>{html.escape(name.capitalize())} matrices</h2>\n")
                df.to_html(f, na_rep="")
                f.write("\n")
            f.write("</body>\n</html>\n")
    elif extension == ".xlsx":
        for name, df in exported.items():
            # the index takes one more column
            if len(df.columns) + 1 > excel_max_columns:
                raise ValueError(
                    f"The table for the {name} matrices has {len(df.columns)} columns, but Excel "
                    f"allows only {excel_max_columns} columns in a sheet. Use the HTML export instead."
                )
        with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
            for name, df in exported.items():
                # freeze the column headers and the matrix names
                df.to_excel(writer, sheet_name=name, freeze_panes=(df.columns.nlevels + 1, 1))
    else:
        raise ValueError(f"Unknown export format of {file_name}, use .html or .xlsx.")
    print(f"Complete tables exported to {file_name}")


def get_flat_df(df, transposed):
    """
    Convert the complete table to the flat (long) form: one row per matrix,
    format, device and launch config, with the columns

      - "matrix name", "transposed", "format", "device", "launch config",
      - the metrics: "time median" (in seconds), "bandwidth" (in TB/s, as in
        the complete table), "diff.max", ...; the speed-ups
        get the name "speed-up vs <reference>", e.g. "speed-up vs cusparse",
        where the reference is given, otherwise just "speed-up" (the
        multi-threaded CPU runs compared to one thread),
      - the properties of the matrix ("rows", "nonzeros", "symmetric", the
        statistics of MatrixStatColumns.py, ...), repeated on all rows of the
        matrix.

    The combinations without any value (e.g. a format which was not run on
    some device) are left out.
    """
    name_column = ("Matrix name", "", "", "", "")
    matrix_columns = [column for column in df.columns if column[1] == "" and column != name_column]
    result_columns = [column for column in df.columns if column[1] != ""]

    matrices = df[matrix_columns].copy()
    matrices.columns = [column[0] for column in matrix_columns]
    matrices.insert(0, "transposed", transposed)
    matrices.insert(0, "matrix name", df[name_column].to_numpy())

    results = df[result_columns].apply(pd.to_numeric, errors="coerce")
    results.columns = pd.MultiIndex.from_tuples(
        [
            (format, device, launch_config, f"{metric} vs {reference}" if reference else metric)
            for format, device, launch_config, metric, reference in result_columns
        ],
        names=["format", "device", "launch config", None],
    )
    results.index = df.index
    flat = results.stack(level=["format", "device", "launch config"], future_stack=True)
    flat = flat.dropna(axis="index", how="all").reset_index(level=["format", "device", "launch config"])
    flat = matrices.join(flat, how="inner")
    return flat.reset_index(drop=True)


def export_flat_tables(file_name, tables):
    """
    Export the complete tables for the original and the transposed matrices in
    the flat form (see get_flat_df()) to one Parquet or CSV file, depending on
    the file extension
    """
    flat = pd.concat(
        [get_flat_df(tables[name]["df"], name == "transposed") for name in table_names if tables.get(name)],
        ignore_index=True,
    )
    extension = os.path.splitext(file_name)[1].lower()
    print(f"Exporting the flat table to {file_name} ...")
    if extension == ".parquet":
        flat.to_parquet(file_name, index=False)
    elif extension == ".csv":
        flat.to_csv(file_name, index=False)
    else:
        raise ValueError(f"Unknown flat export format of {file_name}, use .parquet or .csv.")
    print(f"Flat table with {len(flat.index)} rows exported to {file_name}")


def print_formats_and_launch_configs(formats, launch_configs, accelerator_devices):
    """
    Print formats and their launch configurations.
    """
    print(f"Formats: {formats}")
    for format in formats:
        for device in ["CPU"] + accelerator_devices:
            if (format, device) in launch_configs:
                print(
                    f"Launch configs for {format} on {device}: {launch_configs[(format, device)]}"
                )
