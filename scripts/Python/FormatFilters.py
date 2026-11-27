"""
Selection of the formats in the complete table written by
tnl-spmv-benchmark-parse-logs.py: the best time per matrix over a group of
formats, the "plain CSR" baseline, the storage families and the formats of
the vendor library and of the other libraries (Ginkgo, Hypre). They are
shared by the analysis tables (AnalysisTables.py) and the poster charts
(poster-sc2026/), so that all of them agree on what "the CSR baseline" means
and on which vendor library each device is compared with.
"""

import pandas as pd

from VendorLibraries import vendor_format


def _min_time_per_matrix(df, device, format_ok):
    """
    Per-matrix minimum of "time median" across all (format, launch_config)
    columns on `device` whose format passes `format_ok` - i.e. the best time
    achievable on that matrix by picking the best launch config (and, when
    format_ok matches several formats, the best format) from the group.

    Returns None if no matching column exists for this device.
    """
    cols = [
        col
        for col in df.columns
        if len(col) == 5 and col[1] == device and col[3] == "time median" and format_ok(col[0])
    ]
    if not cols:
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").min(axis=1)


# ---------------------------------------------------------------------------
# "Best plain CSR" baseline: the reference every chart module compares
# against, in one place so the definition of "plain CSR" (and the set of
# storage families) can't drift between modules.
# ---------------------------------------------------------------------------

# "Plain" CSR variants (no Binary/Symmetric compression, which change the
# problem/data assumptions) - these form the "best CSR" half of the baseline
# used by PosterHeatmapGraphs.speedup_heatmap_vs_best_csr() and
# PosterCoverageGraphs.cumulative_coverage_vs_best(): CSR, Adaptive CSR,
# Sorted CSR, Sorted Adaptive CSR and their launch configs.
CSR_BASELINE_EXCLUDE_TAGS = ("Binary", "Symmetric", "Legacy", "Best")

# One row/step per storage family. RowMajor/ColumnMajor SlicedEllpack fold
# their slice-size suffix (e.g. "RowMajor SlicedEllpack 16") into one entry
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


# ---------------------------------------------------------------------------
# Vendor sparse library per accelerator device (see VendorLibraries.py).
# "Best vendor" time is the minimum over every format/algorithm the library
# offers on that device (e.g. cuSPARSE CSR with all its algorithms and
# cuSPARSE SlicedEll).
# ---------------------------------------------------------------------------

def _vendor_format_filter(device):
    """
    Predicate matching every format of the vendor library on `device`, or
    None if there is no known vendor library for that device.
    """
    prefix = vendor_format(device)
    if prefix is None:
        return None
    return lambda format: format.startswith(prefix)


# ---------------------------------------------------------------------------
# The other libraries (Ginkgo, Hypre). Hypre has only the CSR format, logged
# as "Hypre". The CSR format of Ginkgo is logged as "Ginkgo" with its SpMV
# strategies as launch configs, the other formats as "Ginkgo Ell", "Ginkgo
# Sellp", "Ginkgo Hybrid" and "Ginkgo Coo". "Ginkgo Sparselib" is the CSR
# format with the strategy which calls the vendor library (cuSPARSE,
# hipSPARSE), so it is not counted as a kernel of Ginkgo. Hypre calls the
# vendor library by default (launch config "Default", its own kernel is
# "Internal"), which cannot be told apart by the format, so both are Hypre.
# ---------------------------------------------------------------------------

GINKGO_VENDOR_FORMAT = "Ginkgo Sparselib"


def _library_format_filter(library, csr_only=False):
    """
    Predicate matching the formats of `library` ("Ginkgo" or "Hypre") which
    are its own kernels: all of them, or only the CSR format with csr_only.
    """
    if csr_only:
        return lambda format: format == library
    return lambda format: (format == library or format.startswith(f"{library} ")) and format != GINKGO_VENDOR_FORMAT


def _best_vendor_time(df, device):
    """
    Per-matrix best time of the vendor library on `device` (best format and
    algorithm), or None if the library has no results on that device.
    """
    vendor_ok = _vendor_format_filter(device)
    if vendor_ok is None:
        return None
    return _min_time_per_matrix(df, device, vendor_ok)
