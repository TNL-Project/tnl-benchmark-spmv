"""
Integrity checks of the table built by tnl-spmv-benchmark-parse-logs.py.

The table is checked against the records of the logs it was built from and
against itself, always in another way than it was computed:

  - the numbers and the sums of the times and the bandwidths of each matrix
    (a row of the table) and of each format, device and launch configuration
    (a column) must agree with the records grouped the same way. The sums over
    the rows reveal the results assigned to a wrong matrix, the sums over the
    columns those assigned to a wrong format, device or launch configuration.
  - the numbers of rows, columns and nonzeros of each matrix must agree with
    the records of that matrix.
  - each speed-up is a ratio of two times (or two bandwidths), so the time
    it was computed from together with the time of the same format, device
    and launch configuration must be one of the times in the table. The times
    found this way are written to the report, which shows what each speed-up
    is really computed against.
  - the parallel efficiency must be the speed-up divided by the number of
    threads.

The records of the logs themselves are checked too, the failures are reported
as warnings:

  - all records should be computed in the same precision.
  - the bandwidth multiplied by the time should be the size of the data of
    the matrix. The benchmark counts the elements before the compression of
    the matrix, so the bounds are wide.
  - no matrix should have much less results than the others, which happens
    e.g. when the same matrix has different names in the logs.
"""

import numpy as np
import pandas as pd

from Speedup import cpu_threads_count

REL_TOL = 1e-9
MAX_LISTED = 10
# bounds of (bandwidth * time) / (size of the matrix data)
DATA_SIZE_RATIO_BOUNDS = (0.3, 1.2)
# a matrix should have at least this fraction of the median number of results
MIN_RESULTS_FRACTION = 0.1

MATRIX_NAME = ("Matrix name", "", "", "", "")
RECORD_KEY = ["format", "performer", "launch cfg."]


class Checks:
    """
    Collects the results of the checks of one table
    """

    def __init__(self, name):
        self.name = name
        self.lines = []
        self.errors = 0
        self.warnings = 0

    def _add(self, status, message, details):
        self.lines.append(f"   {status:8}{message}")
        details = list(details)
        for detail in details[:MAX_LISTED]:
            self.lines.append(f"              {detail}")
        if len(details) > MAX_LISTED:
            self.lines.append(f"              ... and {len(details) - MAX_LISTED} more")

    def ok(self, message, details=()):
        self._add("OK", message, details)

    def info(self, message, details=()):
        self._add("INFO", message, details)

    def warning(self, message, details=()):
        self.warnings += 1
        self._add("WARNING", message, details)

    def error(self, message, details=()):
        self.errors += 1
        self._add("ERROR", message, details)

    def result(self, failed, message, details=(), warning=False):
        """
        Add an error (or a warning) with the details if `failed`, OK otherwise
        """
        if not failed:
            self.ok(message)
        elif warning:
            self.warning(message, details)
        else:
            self.error(message, details)

    def report(self):
        header = f"Integrity checks of the table for the {self.name} matrices: "
        header += f"{self.errors} errors, {self.warnings} warnings"
        return "\n".join([header] + self.lines)

    def summary(self):
        """
        The header with the errors and the warnings only
        """
        lines = self.report().split("\n")
        keep = False
        result = [lines[0]]
        for line in lines[1:]:
            if not line.startswith("              "):
                keep = line.startswith("   ERROR") or line.startswith("   WARNING")
            if keep:
                result.append(line)
        return "\n".join(result)


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _close(a, b):
    """
    Elementwise comparison with the relative tolerance, NaN equals NaN
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.isclose(a, b, rtol=REL_TOL, atol=0.0) | (np.isnan(a) & np.isnan(b))


def check_matrices(df, records, checks):
    names = df[MATRIX_NAME]
    duplicated = names[names.duplicated()].tolist()
    checks.result(duplicated, f"The names of the {len(names)} matrices in the table are unique", duplicated)
    record_names = set(records["matrix name"])
    unknown = sorted(set(names) - record_names)
    checks.result(unknown, "All matrices in the table are in the logs", unknown)
    missing = sorted(record_names - set(names))
    if missing:
        checks.info(f"{len(missing)} matrices of the logs are not in the table (--max-rows)")


def check_matrix_properties(df, records, checks):
    names = df[MATRIX_NAME].values
    for column in ["rows", "columns", "nonzeros"]:
        if column not in records.columns:
            continue
        groups = _numeric(records[column]).groupby(records["matrix name"])
        expected = groups.first().reindex(names).values
        table = _numeric(df[(column, "", "", "", "")]).values
        wrong = ~_close(table, expected)
        checks.result(
            wrong.any(),
            f"The number of {column} of each matrix agrees with the logs",
            (f"{name}: {t:g} in the table, {e:g} in the logs" for name, t, e in zip(names[wrong], table[wrong], expected[wrong])),
        )
        different = groups.nunique()
        different = different[different > 1].index.intersection(names)
        checks.result(
            len(different),
            f"The number of {column} of each matrix is the same in all its records",
            different,
            warning=True,
        )
    rows = _numeric(df[("rows", "", "", "", "")]).values
    nonzeros = _numeric(df[("nonzeros", "", "", "", "")]).values
    per_row = _numeric(df[("nonzeros per row", "", "", "", "")]).values
    wrong = ~_close(per_row * rows, nonzeros)
    checks.result(wrong.any(), "nonzeros = rows * nonzeros per row", names[wrong])


def check_values(df, records, checks, record_column, metric, factor=1.0):
    """
    Compare the numbers and the sums of the values of the metric in the rows
    and in the columns of the table with the records
    """
    names = df[MATRIX_NAME].values
    records = records[records["matrix name"].isin(set(names))]
    values = _numeric(records[record_column]) * factor
    records = records.loc[values.notna(), ["matrix name"] + RECORD_KEY].copy()
    records["value"] = values[values.notna()]

    table_columns = [column for column in df.columns if column[3] == metric and column[4] == ""]
    table = df[table_columns].apply(_numeric).to_numpy(dtype=float)
    keys = {column[:3] for column in table_columns}
    in_table = np.array([key in keys for key in zip(*(records[k] for k in RECORD_KEY))], dtype=bool)
    if not in_table.all():
        left_out = records[~in_table].groupby("format").size()
        checks.info(
            f"{(~in_table).sum()} records with the {metric} are of formats which are not in the table",
            (f"{format}: {count} records" for format, count in left_out.items()),
        )
    records = records[in_table]

    table_count = int((~np.isnan(table)).sum())
    checks.result(
        table_count != len(records.index),
        f"The table has as many values of the {metric} as the logs ({len(records.index)})",
        [f"{table_count} in the table"],
    )

    by_matrix = records.groupby("matrix name")["value"].agg(["count", "sum"]).reindex(names)
    wrong = (by_matrix["count"].fillna(0).values != (~np.isnan(table)).sum(axis=1)) | ~_close(
        by_matrix["sum"].fillna(0.0).values, np.nansum(table, axis=1)
    )
    checks.result(
        wrong.any(),
        f"The number and the sum of the values of the {metric} of each matrix agree with the logs",
        names[wrong],
    )

    by_column = records.groupby(RECORD_KEY)["value"].agg(["count", "sum"])
    by_column = by_column.reindex(pd.MultiIndex.from_tuples([column[:3] for column in table_columns]))
    wrong = (by_column["count"].fillna(0).values != (~np.isnan(table)).sum(axis=0)) | ~_close(
        by_column["sum"].fillna(0.0).values, np.nansum(table, axis=0)
    )
    checks.result(
        wrong.any(),
        f"The number and the sum of the values of the {metric} of each format, device and launch "
        "configuration agree with the logs",
        (" / ".join(column[:3]) for column, w in zip(table_columns, wrong) if w),
    )


def _find_columns(values, implied, skip):
    """
    Return the indexes of the columns of `values` which agree with `implied`
    in all rows where it is defined. Zeros (e.g. a bandwidth which the
    benchmark did not compute) agree with nothing.
    """
    valid = ~np.isnan(implied)
    if (implied[valid] == 0).any():
        return []
    first = np.argmax(valid)
    candidates = np.nonzero(np.isclose(values[first, :], implied[first], rtol=REL_TOL, atol=0.0))[0]
    return [
        j
        for j in candidates
        if j != skip and np.isclose(values[valid, j], implied[valid], rtol=REL_TOL, atol=0.0).all()
    ]


def check_speedups(df, checks):
    """
    Check that each speed-up is a ratio of two times (or bandwidths) in the
    table and report what the speed-ups are computed against
    """
    # metric -> (columns, their indexes by (format, device, launch config), values)
    metrics = {}
    for metric, name in [("time median", "time"), ("bandwidth", "bandwidth")]:
        columns = [column for column in df.columns if column[3] == metric]
        index = {column[:3]: j for j, column in enumerate(columns)}
        metrics[name] = (columns, index, df[columns].apply(_numeric).to_numpy(dtype=float))

    empty = 0
    checked = 0
    invalid = []
    without_time = []
    unmatched = []
    # (device, metric, reference) -> {"columns": count, "definitions": set of descriptions}
    definitions = {}
    for column in df.columns:
        if column[3] not in ["speed-up", "TNL speed-up"]:
            continue
        speedup = _numeric(df[column]).to_numpy(dtype=float)
        defined = ~np.isnan(speedup)
        if not defined.any():
            empty += 1
            continue
        checked += 1
        label = " / ".join(part for part in column if part)
        if not (np.isfinite(speedup[defined]) & (speedup[defined] > 0)).all():
            invalid.append(label)
        _, time_index, times = metrics["time"]
        own = time_index.get(column[:3])
        if own is None or np.isnan(times[defined, own]).any():
            without_time.append(label)
            continue
        group = definitions.setdefault((column[1], column[3], column[4]), {"columns": 0, "definitions": set()})
        group["columns"] += 1
        found = False
        # the bandwidths are tried only for the speed-ups which are not computed from the times
        for name, (columns, index, values) in metrics.items():
            own = index.get(column[:3])
            if own is None or found:
                continue
            # the speed-up is either other / own or own / other
            for implied, description in [
                (speedup * values[:, own], "{other} / own " + name),
                (values[:, own] / speedup, "own " + name + " / {other}"),
            ]:
                for j in _find_columns(values, implied, own):
                    other = columns[j][:3]
                    # the same format and device with another launch configuration is not named in full
                    if other[:2] == column[:2]:
                        other_name = f"{name} of the same format with '{other[2]}'"
                    else:
                        other_name = f"{name} of " + " / ".join(other)
                    group["definitions"].add(description.format(other=other_name))
                    found = True
        if not found:
            unmatched.append(label)

    checks.result(invalid, f"All values of the {checked} speed-ups are positive and finite", invalid)
    checks.result(
        without_time,
        "The time of the format, device and launch configuration is known wherever its speed-up is",
        without_time,
    )
    checks.result(
        unmatched,
        "Each speed-up is a ratio of two different times or bandwidths in the table",
        unmatched,
    )
    if empty:
        checks.info(f"{empty} speed-ups have no values")
    for (device, metric, reference), group in sorted(definitions.items()):
        name = f"{metric} on {device}" + (f" w.r.t. {reference}" if reference else "")
        found = sorted(group["definitions"])
        if len(found) == 1:
            checks.info(f"{name} ({group['columns']} columns) = {found[0]}")
        else:
            checks.info(
                f"{name} ({group['columns']} columns) is computed from {len(found)} different columns, e.g.",
                found[:3],
            )


def check_efficiency(df, checks):
    wrong = []
    count = 0
    for column in df.columns:
        if column[3] != "eff.":
            continue
        threads = cpu_threads_count(column[2])
        speedup_column = (column[0], column[1], column[2], "speed-up", "")
        if threads is None or speedup_column not in df.columns:
            wrong.append(" / ".join(column[:3]))
            continue
        count += 1
        efficiency = _numeric(df[column]).values
        speedup = _numeric(df[speedup_column]).values
        if not _close(efficiency * threads, speedup).all():
            wrong.append(" / ".join(column[:3]))
    checks.result(wrong, f"The {count} parallel efficiencies are the speed-ups divided by the number of threads", wrong)


def check_precision(records, checks):
    if "precision" not in records.columns:
        return
    counts = records.groupby(["precision", "performer"]).size()
    checks.result(
        records["precision"].nunique() > 1,
        "All records are computed in the same precision",
        (f"{precision} on {performer}: {count} records" for (precision, performer), count in counts.items()),
        warning=True,
    )


def check_data_size(records, checks):
    """
    Check that bandwidth * time of each record is the size of the matrix data
    as the benchmark defines it, i.e. 2 reals and 1 index per nonzero. The
    bandwidth in the logs is in GB/s and it is computed from the mean time.
    """
    time = _numeric(records["time"] if "time" in records.columns else records["time median"])
    bandwidth = _numeric(records["bandwidth"])
    real_bytes = pd.Series(8, index=records.index)
    if "precision" in records.columns:
        real_bytes = records["precision"].map({"float": 4, "double": 8}).fillna(8)
    expected = _numeric(records["nonzeros"]) * (2 * real_bytes + 4) / 2**30
    ratio = bandwidth * time / expected
    lower, upper = DATA_SIZE_RATIO_BOUNDS
    wrong = ratio.notna() & ~((ratio >= lower) & (ratio <= upper))
    groups = records[wrong].assign(ratio=ratio[wrong]).groupby(["format", "performer"])["ratio"]
    checks.result(
        wrong.any(),
        f"bandwidth * time of each record is {lower} to {upper} of the size of the matrix data",
        (
            f"{format} / {performer}: {count:.0f} records, median ratio {median:.3g}"
            for (format, performer), (count, median) in groups.agg(["count", "median"]).iterrows()
        ),
        warning=True,
    )


def check_results_count(df, checks):
    names = df[MATRIX_NAME].values
    time_columns = [column for column in df.columns if column[3] == "time median"]
    counts = df[time_columns].apply(_numeric).notna().sum(axis=1).values
    limit = MIN_RESULTS_FRACTION * np.median(counts)
    few = counts < limit
    checks.result(
        few.any(),
        f"Each matrix has at least {MIN_RESULTS_FRACTION:.0%} of the median number of results ({np.median(counts):.0f})",
        (f"{name}: {count} results" for name, count in zip(names[few], counts[few])),
        warning=True,
    )


def check_table(name, table, records, bandwidth_factor=1.0):
    """
    Run all checks of the table (as returned by build_table()) built from the
    records and return the Checks with the results
    """
    checks = Checks(name)
    df = table["df"]
    check_matrices(df, records, checks)
    check_matrix_properties(df, records, checks)
    check_values(df, records, checks, "time median", "time median")
    check_values(df, records, checks, "bandwidth", "bandwidth", bandwidth_factor)
    check_speedups(df, checks)
    check_efficiency(df, checks)
    check_precision(records, checks)
    check_data_size(records, checks)
    check_results_count(df, checks)
    return checks
