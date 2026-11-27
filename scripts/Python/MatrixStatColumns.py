"""
Names of the per-matrix statistics written by tnl-benchmark-spmv-matrix-info
(see include/matrix-info.h and include/MatrixStatistics.h), which
tnl-spmv-benchmark-parse-logs.py copies to the complete table.
"""

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
