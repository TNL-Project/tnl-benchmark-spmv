# Shared by run-tnl-benchmark-spmv and run-tnl-benchmark-spmv-matrix-info: derive a
# system-independent matrix name from its path, e.g.
# ./mtx_matrices/suitesparse-collection-website.herokuapp.com/MM/ANSYS/Delor64K/Delor64K.mtx
# becomes ANSYS/Delor64K.mtx.
#
# Expects the caller to have set `matrices_base_path`.
matrix_name_from_path() {
    local name="${1#"${matrices_base_path%/}"/}"
    name="${name##/}"
    # strip the prefix of known matrix collections
    case "$name" in
        */MM/*)              name="${name##*/MM/}" ;;               # SuiteSparse
        */MatrixMarket2/*)   name="${name##*/MatrixMarket2/}" ;;    # Matrix Market (NIST)
    esac
    # SuiteSparse archives unpack into a directory named after the matrix,
    # so Group/Name/Name.mtx is collapsed into Group/Name.mtx
    local file dir
    file="$(basename "$name")"
    dir="$(dirname "$name")"
    if [[ "$(basename "$dir")" == "${file%.mtx}" ]]; then
        name="$(dirname "$dir")/$file"
        name="${name#./}"
    fi
    echo "$name"
}
