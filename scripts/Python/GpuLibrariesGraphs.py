"""
Comparison of the SpMV on the GPU computed by TNL, the vendor library of the
device (cuSPARSE on CUDA, hipSPARSE on HIP), Ginkgo and Hypre by the
Dolan-More performance profiles (see CpuLibrariesGraphs.py).

The time of a library on a matrix is the best time over the formats allowed
by the variant and all their launch configurations (the threads per segment
of TNL, the algorithms of cuSPARSE), i.e. a per-matrix oracle best:

  - best CSR: all libraries use the CSR format. TNL chooses from CSR and
    Adaptive CSR, both with the rows in the original order and sorted by
    their length (the same formats as the "best CSR" baseline of the other
    poster graphs), the vendor library uses only its CSR format.
  - best available format: TNL chooses from all its formats and the vendor
    library from all its benchmarked formats (cuSPARSE CSR and SlicedEll,
    hipSPARSE CSR only). Ginkgo and Hypre are benchmarked only with one
    format and configuration, so their times are the same as in the best CSR
    variant and they are not the best the library can do.

The binary and symmetric formats of TNL are left out, they need a matrix with
unit values or a symmetric one. A library which is not benchmarked on a device
(Hypre on CUDA) is missing in the panel of that device.

Both devices are drawn side by side with the same axes and for the same
matrices - those with the results of all libraries on all devices. The other
matrices are listed in a text file.
"""

import os
import pandas as pd

from VendorLibraries import vendor_label, vendor_format, is_vendor_format
from PosterGraphsCommon import _min_time_per_matrix, _is_csr_baseline_format
from CpuLibrariesGraphs import MATRIX_NAME, performance_ratios, draw_performance_profiles

# formats of the other libraries, they are benchmarked with one configuration
OTHER_LIBRARIES = ["Ginkgo", "Hypre"]
# libraries in the table which are not compared
IGNORED_LIBRARIES = ("LightSpMV",)

# TNL formats left out: they do not work for a general matrix ("Binary",
# "Symmetric") or they are not formats of the current TNL ("Legacy", "Best")
TNL_EXCLUDE_TAGS = ("Binary", "Symmetric", "Legacy", "Best")

# the colors of TNL, Hypre and Ginkgo are the same as in CpuLibrariesGraphs.py
LIBRARY_STYLES = {
    "TNL": {"color": "#2a78d6", "linestyle": "-"},
    "Hypre": {"color": "#eb6834", "linestyle": "--"},
    "Ginkgo": {"color": "#1baf7a", "linestyle": ":"},
}
VENDOR_STYLE = {"color": "#8b5cc7", "linestyle": "-."}

# The largest ratio shown in the graphs and the ticks of the axis. The largest
# ratios are in hundreds, which would squeeze the interesting part of the
# profile, all ratios are in the CSV files.
TAU_LIMIT = 8
TAU_TICKS = [1, 1.5, 2, 3, 4, 6, 8]


def _is_tnl_format(format):
    return (
        not is_vendor_format(format)
        and not format.startswith(tuple(OTHER_LIBRARIES) + IGNORED_LIBRARIES)
        and not any(tag in format for tag in TNL_EXCLUDE_TAGS)
    )


# variant -> (title, formats of TNL, formats of the vendor library with the given prefix)
VARIANTS = {
    "best-csr": (
        "best CSR of each library",
        _is_csr_baseline_format,
        lambda prefix: lambda format: format == prefix,
    ),
    "best-format": (
        "best benchmarked format of each library",
        _is_tnl_format,
        lambda prefix: lambda format: format.startswith(prefix),
    ),
}


def collect_times(df, device, variant):
    """
    Return a dataframe indexed by the matrix name with the best time of each
    library benchmarked on the device (NaN where it is missing)
    """
    _, tnl_ok, vendor_ok = VARIANTS[variant]
    filters = {"TNL": tnl_ok}
    if vendor_format(device) is not None:
        filters[vendor_label(device)] = vendor_ok(vendor_format(device))
    for library in OTHER_LIBRARIES:
        filters[library] = lambda format, library=library: format == library

    times = pd.DataFrame(index=df[MATRIX_NAME].values)
    for label, format_ok in filters.items():
        best = _min_time_per_matrix(df, device, format_ok)
        if best is not None and best.notna().any():
            times[label] = best.values
    return times


def used_formats(df, device, variant):
    """
    Return the formats of TNL and the vendor library on the device over which
    the best time is taken
    """
    _, tnl_ok, vendor_ok = VARIANTS[variant]
    formats = sorted({column[0] for column in df.columns if len(column) == 5 and column[1] == device})
    result = {"TNL": [format for format in formats if tnl_ok(format)]}
    if vendor_format(device) is not None:
        prefix_ok = vendor_ok(vendor_format(device))
        result[vendor_label(device)] = [format for format in formats if prefix_ok(format)]
    return result


def write_missing_matrices(file_name, times_by_panel, common):
    """
    Write the numbers of the matrices with the results of each library and the
    matrices left out of the comparison with the libraries missing them
    """
    lines = [f"Matrices with the results of all libraries on all devices: {len(common)}"]
    for (device, variant), times in times_by_panel.items():
        counts = ", ".join(f"{label} {times[label].notna().sum()}" for label in times.columns)
        lines.append(f"   {device}, {variant}: {counts} of {len(times.index)}")
    print("\n".join(lines))

    missing = {}
    for (device, variant), times in times_by_panel.items():
        for label in times.columns:
            for matrix in times.index[times[label].isna()]:
                missing.setdefault(matrix, set()).add(f"{label} ({device})")
    with open(file_name, "w") as file:
        file.write("\n".join(lines) + "\n\nMatrices left out:\n")
        for matrix in sorted(missing):
            file.write(f"   {matrix:60} missing: {', '.join(sorted(missing[matrix]))}\n")
    print(f"The matrices missing for some library are listed in {file_name}")


def gpu_libraries_graphs(df, accelerator_devices, transposed=False, output_dir="GPU-libraries"):
    """
    Entry point: write the performance profiles of TNL, the vendor library,
    Ginkgo and Hypre on the GPUs for the best CSR and the best format of each
    library, the ratios to the best time as CSV, the formats used and the list
    of the matrices left out
    """
    if not accelerator_devices:
        print("No GPU results, skipping the GPU libraries graphs.")
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    suffix = "-transposed" if transposed else ""
    matrices = "transposed matrices" if transposed else "original matrices"

    times_by_panel = {
        (device, variant): collect_times(df, device, variant)
        for variant in VARIANTS
        for device in accelerator_devices
    }
    times_by_panel = {panel: times for panel, times in times_by_panel.items() if len(times.columns) > 1}
    if not times_by_panel:
        print("No GPU results of the other libraries, skipping the GPU libraries graphs.")
        return
    # the same matrices in all panels
    common = None
    for times in times_by_panel.values():
        complete = times.dropna().index
        common = complete if common is None else common.intersection(complete)
    write_missing_matrices(os.path.join(output_dir, f"missing-matrices{suffix}.txt"), times_by_panel, common)
    if common.empty:
        print("No matrices with the results of all libraries on all devices.")
        return

    with open(os.path.join(output_dir, f"formats{suffix}.txt"), "w") as file:
        for (device, variant), times in times_by_panel.items():
            file.write(f"{device}, {VARIANTS[variant][0]}:\n")
            for label, formats in used_formats(df, device, variant).items():
                file.write(f"   {label}: {', '.join(formats)}\n")

    # the vendor libraries of all devices share one item of the legend
    devices = list(dict.fromkeys(device for device, _ in times_by_panel))
    vendors_label = " / ".join(vendor_label(device) for device in devices if vendor_label(device))

    for variant, (variant_title, _, _) in VARIANTS.items():
        ratios_by_device = {
            device: performance_ratios(times.loc[common])
            for (device, panel_variant), times in times_by_panel.items()
            if panel_variant == variant
        }
        csv = pd.concat(ratios_by_device, axis=1, names=["device", "library"])
        csv.index.name = "matrix"
        csv.to_csv(os.path.join(output_dir, f"ratios-to-best-{variant}{suffix}.csv"))
        print(f"Writing the GPU libraries performance profiles for the {variant_title}")
        draw_performance_profiles(
            {
                device: ratios.rename(columns={vendor_label(device): vendors_label})
                for device, ratios in ratios_by_device.items()
            },
            os.path.join(output_dir, f"performance-profiles-{variant}{suffix}.pdf"),
            title=f"Performance profiles of SpMV on the GPU, {variant_title}, {matrices}",
            panel_size=(5.0, 3.8),
            styles={**LIBRARY_STYLES, vendors_label: VENDOR_STYLE},
            tau_limit=TAU_LIMIT,
            tau_ticks=TAU_TICKS,
        )
