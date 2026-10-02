"""
Maps a (new format, launch config) to the name of the "legacy" (pre-rewrite)
TNL implementation it replaces, so a speed-up column comparing the two can be
added - i.e. "did the rewrite actually help?".

Shared by the speed-up computation (Speedup.py, called from
tnl-spmv-benchmark-parse-logs.py) and the HTML report (Report.py, called from
tnl-spmv-benchmark-make-tables-json.py).
"""

legacy_counterparts = {
    ("BiEllpack", "1 TPS"): "Legacy BiEllpack",
    ("ChunkedEllpack", "1 TPS"): "Legacy ChunkedEllpack",
    ("SlicedEllpack", "1 TPS"): "Legacy SlicedEllpack",
    ("Ellpack", "1 TPS"): "Legacy Ellpack",
    ("CSR", "1 TPS"): "Legacy CSR Scalar",
    ("CSR", "Warp per segment"): "Legacy CSR Vector",
    ("CSR", "Light CSR"): "Legacy CSR LightWithoutAtomic",
    ("CSR Adaptive", "Default"): "Legacy CSR Adaptive",
}
