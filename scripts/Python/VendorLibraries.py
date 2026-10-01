"""
Vendor sparse library of each accelerator device. A device's results are
only ever compared with the vendor library on that same device - CUDA with
cuSPARSE, HIP with hipSPARSE - never across devices.

Shared by the speed-up computation (Speedup.py,
tnl-spmv-benchmark-make-tables-json.py) and the HTML report (Report.py,
BestFormats.py), so they all agree on which library is "the vendor library"
of a device.
"""

# device -> (label, format): `format` is the name the reference benchmark
# logs the library's CSR SpMV under, and also the prefix of every other
# format of that library (e.g. "cusparse SlicedEll").
VENDOR_LIBRARIES = {
    "CUDA": ("cuSPARSE", "cusparse"),
    "HIP": ("hipSPARSE", "hipsparse"),
}


def vendor_label(device):
    """Display name of the vendor library on `device`, or None if unknown."""
    vendor = VENDOR_LIBRARIES.get(device)
    return vendor[0] if vendor else None


def vendor_format(device):
    """
    Format name of the vendor library's CSR SpMV on `device` - also the
    "reference" name of the speed-up columns computed against it - or None
    if there is no known vendor library for that device.
    """
    vendor = VENDOR_LIBRARIES.get(device)
    return vendor[1] if vendor else None


def is_vendor_format(format):
    """True for every format of any vendor library."""
    return any(format.startswith(prefix) for _, prefix in VENDOR_LIBRARIES.values())
