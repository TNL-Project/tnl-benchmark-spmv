# tnl-benchmark-spmv

Benchmarks for the sparse matrix–vector multiplication (SpMV) with matrix
formats in TNL.

## Getting started

1. Install [Git](https://git-scm.com/).

2. Clone the repository, making sure that Git submodules are initialized:

       git clone --recurse-submodules <this_repo_url>

   If you cloned the repository without the `--recurse-submodules` option,
   the submodules can be initialized subsequently:

       git submodule update --init --recursive

3. Install the necessary tools and dependencies:

    - [CMake](https://cmake.org/) build system (version 3.24 or newer)
    - [CUDA](https://docs.nvidia.com/cuda/index.html) toolkit (version 11 or
      newer)
    - compatible host compiler (e.g. [GCC](https://gcc.gnu.org/) or
      [Clang](https://clang.llvm.org/))

4. Configure the build using `cmake` in the root path of the Git repository:

       cmake -B build -S . <additional_configure_options...>

   This will use `build` in the current path as the build directory.
   The path for the `-S` option corresponds to the root path of the project.
   You may use additional options to configure the build:

   - `-DCMAKE_BUILD_TYPE=<type>` where `<type>` is one of `Debug`, `Release`,
     `RelWithDebInfo`
   - `-DCMAKE_CUDA_ARCHITECTURES=<arch>` – to build for a CUDA architecture
     other than "native"
   - `-DSYSTEM_TNL` – to use TNL installed on the system instead of the Git
     submodule

5. Build the targets using `cmake`:

       cmake --build build

6. Edit the launcher script [scripts/run-benchmark](./scripts/run-benchmark) as
   appropriate and run it:

       ./scripts/run-benchmark
