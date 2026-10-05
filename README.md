# tnl-benchmark-spmv

Benchmarks for the sparse matrix–vector multiplication (SpMV) with matrix
formats in [TNL](https://gitlab.com/tnl-project/tnl), compared with the
vendor and third-party libraries (cuSPARSE, hipSPARSE, Hypre, Ginkgo,
LightSpMV).

## Building

1. Install [Git](https://git-scm.com/) and clone the repository:

       git clone <this_repo_url>

2. Install the necessary tools and dependencies:

    - [CMake](https://cmake.org/) build system (version 3.24 or newer)
    - a C++17 host compiler (e.g. [GCC](https://gcc.gnu.org/) or
      [Clang](https://clang.llvm.org/))
    - for GPU benchmarks: the [CUDA](https://docs.nvidia.com/cuda/index.html)
      toolkit or [ROCm](https://rocm.docs.amd.com/) with hipSPARSE
    - optionally MPI with [Hypre](https://github.com/hypre-space/hypre)
      and/or [Ginkgo](https://ginkgo-project.github.io/) for the reference
      benchmarks

   TNL itself is downloaded by CMake during the configuration (branch
   `main`).

3. Configure the build in the root path of the Git repository:

       cmake -B build -S . -DCMAKE_BUILD_TYPE=Release <additional_options...>

   Useful options:

   - `-DCMAKE_CUDA_ARCHITECTURES=<arch>` – build for a CUDA architecture
     other than "native"
   - `-DTNL_USE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=<arch>` – build for HIP
     (e.g. `gfx90a`, `gfx1100`). CUDA takes precedence when both are
     available, so CUDA and HIP need separate build directories.
   - `-DTNL_ENABLE_NATIVE=ON` or `-DTNL_CPU_ARCH=<arch>` – optimize the host
     code for the native or the given CPU architecture
   - `-DHYPRE_ROOT=<path>`, `-DGinkgo_DIR=<path>` – locations of Hypre and
     Ginkgo. Hypre is used only by the binary matching the backend Hypre
     was built for (CPU, CUDA or HIP).
   - `-DFETCHCONTENT_SOURCE_DIR_TNL=<path>` – use a local TNL source tree
     instead of downloading it

   The presets in [CMakePresets.json](./CMakePresets.json) (`cluster`,
   `local-native`) can be used instead, e.g. `cmake --preset cluster`.

4. Build the targets:

       cmake --build build

   The executables are placed in `build/bin`.

## Benchmark binaries

| binary | what it benchmarks |
|---|---|
| `tnl-benchmark-spmv` | TNL formats on the CPU |
| `tnl-benchmark-spmv-cuda`, `tnl-benchmark-spmv-hip` | TNL formats on a CUDA or HIP GPU |
| `tnl-benchmark-spmv-reference` | Hypre and Ginkgo on the CPU |
| `tnl-benchmark-spmv-reference-cuda` | cuSPARSE (CSR and SlicedEll), LightSpMV, Hypre and Ginkgo on a CUDA GPU |
| `tnl-benchmark-spmv-reference-hip` | hipSPARSE, Hypre and Ginkgo on a HIP GPU |
| `tnl-benchmark-spmv-legacy`, `tnl-benchmark-spmv-legacy-cuda` | the legacy (pre-segments) TNL formats |

The TNL formats are CSR (with all kernels), Ellpack, SlicedEllpack
(row- and column-major, slice sizes 2–32), ChunkedEllpack and BiEllpack,
each also in variants with sorted segments, Binary (matrix values ignored)
and Symmetric (only for symmetric matrices).

Every binary benchmarks one matrix per run, the original and by default
also the transposed one (except the legacy benchmarks). The transposition
is skipped for structurally symmetric matrices, i.e. matrices declared as
symmetric in the MTX file or with a symmetric pattern of the nonzero
elements, since it has the same pattern. Each result is one JSON line
appended to the log file, so the logs of many runs (and binaries) can be
concatenated. The metadata of each record include the matrix name and
statistics, whether the matrix is `symmetric` (declared in the MTX file)
and `structurally symmetric`, the format, the device (`performer`), the
launch configuration, the `build` (`host`, `cuda` or `hip`), the time, the
bandwidth and the difference from the result of CSR on the CPU
(`CSR Diff.Max`, `CSR Diff.L2`).

The CPU runs are labeled with the real number of threads (`1 thread`,
`N threads`). CSR on the CPU is always benchmarked with 1, 2, 4, … threads
up to the maximal number, since it is the reference for the speed-ups.
The GPU builds have no OpenMP and usually run on a different machine, so
they skip the CPU by default.

Basic command line options (see `--help` for all of them):

| option | meaning |
|---|---|
| `--input-file <file.mtx>` | the matrix in the Matrix Market format (required) |
| `--matrix-name <name>` | matrix name written to the log (default: the input file name) |
| `--log-file <file>`, `--output-mode append\|overwrite` | the log file and whether to append to it |
| `--precision float\|double\|all` | precision of the computation (default: `double`) |
| `--with-transposed-matrix true\|false\|only` | benchmark also / not / only the transposed matrix (never for structurally symmetric matrices) |
| `--with-symmetric-matrices`, `--with-sorted-segments`, `--with-sigma-256-sorted-segments`, `--with-ellpack-formats` | enable or disable groups of formats |
| `--with-cpu-tests` | benchmark the CPU (default: yes in the host build, no in the GPU builds) |
| `--with-all-cpu-tests` | on the CPU, benchmark all formats, not only CSR, Binary and Symmetric |
| `--cpu-threads-sweep` | benchmark also the other formats than CSR on the CPU with 1, 2, 4, … threads |
| `--openmp-enabled`, `--openmp-max-threads` | OpenMP settings for the CPU |
| `--loops`, `--min-time`, `--warmup-loops`, `--warmup-min-time` | number of repetitions and warm-up of every measurement |

For example:

    build/bin/tnl-benchmark-spmv-cuda --input-file matrix.mtx --log-file spmv.log --output-mode append

## Getting the matrices

[scripts/Python/get-matrices.py](./scripts/Python/get-matrices.py)
downloads matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/):

    ./scripts/Python/get-matrices.py --max-gpu-memory 16G --output-dir ./mtx_matrices

It downloads the collection's metadata (`ssstats.csv`, cached in the
output directory) and selects the real matrices whose CSR representation
plus the vectors fit into the given GPU memory and whose dimensions fit
into the index type. The selection is written to `selected-matrices.tsv`.
Each selected matrix is then downloaded and extracted to
`<output-dir>/suitesparse/<group>/<name>.mtx`, including the additional
matrices of the archive but without the right-hand sides (`*_b.mtx`).
Interrupted downloads are resumed and already extracted matrices are
skipped, so the script can simply be run again.

Further options: `--scalar-bytes 4|8` and `--index-bytes 4|8` for the
memory estimate, `--include-complex`, `--keep-archives`,
`--no-extract-matrices`, `--refresh-metadata` and `--dry-run` (only select
the matrices and write the manifest).

## Running the benchmarks on many matrices

[scripts/run-tnl-benchmark-spmv](./scripts/run-tnl-benchmark-spmv) runs
one benchmark binary on all `*.mtx` files found recursively in a directory:

    ./scripts/run-tnl-benchmark-spmv --benchmark build/bin/tnl-benchmark-spmv-cuda \
        --input-dir ./mtx_matrices --log-file log-files/sparse-matrix-benchmark-cuda.log

The list of the remaining matrices is kept in the file `input-files` in the
current directory, so an interrupted run can be resumed with
`--continue yes` (otherwise the log file is deleted first). A crash of the
benchmark on one matrix is recorded in `segfaults.log` and the run goes on
with the next matrix. The matrix name written to the log is derived from
the path relative to the input directory, so that logs from different
machines can be merged. Other options limit the size (`--size-limit`, in
bytes) and the number (`--count-limit`) of the matrices, set OpenMP
(`--openmp-enabled`, `--openmp-max-threads`, default: the number of
physical cores) or run the benchmark via `mpirun` (`--mpi-enabled`,
`--mpi-processes`), see `--help`.

[scripts/run-all-benchmarks](./scripts/run-all-benchmarks) is an example
which runs `run-tnl-benchmark-spmv` for several binaries one after another,
each with its own log file in `./log-files`. Edit the paths and limits at
its beginning as appropriate.

## Processing the results

The results are processed in two steps by the Python scripts in
[scripts/Python](./scripts/Python). They need pandas, NumPy, Matplotlib,
openpyxl (for the Excel export) and the `TNL` Python package (`src/Python`
in the TNL repository, installed together with TNL).

1. [tnl-spmv-benchmark-parse-logs.py](./scripts/Python/tnl-spmv-benchmark-parse-logs.py)
   parses the log files and builds one table for the original and one for
   the transposed matrices, with one row per matrix and one column per
   format, device, launch configuration and metric. It also computes the
   speed-ups (compared to the vendor library, CSR on the CPU, Hypre,
   Ginkgo, the non-Binary/non-Symmetric/non-sorted variants and the legacy
   formats) and the parallel efficiency of the multi-threaded CPU runs.
   Both tables are stored in a binary file, which takes most of the
   processing time:

       ./tnl-spmv-benchmark-parse-logs.py -i log-files/*.log -o spmv-benchmark-table.pkl --export spmv-table.xlsx

   - `--export <file.html|file.xlsx>` – export the complete tables also to
     HTML or Excel (one sheet per table)
   - `--duplicates first|last|min|max` – which record to keep when the same
     benchmark appears more than once in the input (default: `first`). The
     CPU results should come from one machine only, the script warns if
     they come from several builds.
   - `--fill-transposed` – fill the table for the transposed matrices with
     the results for the original ones for the structurally symmetric
     matrices, whose transposition is not benchmarked
   - `--max-rows <n>` – process only the first `n` matrices

2. [tnl-spmv-benchmark-make-tables-json.py](./scripts/Python/tnl-spmv-benchmark-make-tables-json.py)
   loads the stored table and writes the reports to the output directory
   (default: `spmv-benchmark-report`):

       ./tnl-spmv-benchmark-make-tables-json.py -i spmv-benchmark-table.pkl -o spmv-benchmark-report --draw-graphs

   - always: the best formats per matrix and their statistics
   - `--draw-graphs` – the full report with graphs of the speed-ups and the
     CPU scalability
   - `--poster-graphs` – the summary charts prepared for a poster
   - `--list-matrices` – the list of processed matrices
   - `--transposed` – process the table for the transposed matrices
