# CSR5 does not work properly yet:
#
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/9
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/10
#
# We can build it with TNL but it crashes with many CUDA errors. We should first check it
# with the original build.
#
#include( cmake/BuildCSR5.cmake )

find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
    find_package(Hypre)
    find_package(Ginkgo)
endif()

# Add the reference libraries (Hypre, Ginkgo) to a reference benchmark target.
# `backend` is one of HOST, CUDA, HIP. Hypre keeps its data in the memory space
# it was built for, so it is added only to the target with the matching backend
# (a CPU build of Hypre to the host target, a CUDA build to the CUDA target and
# a HIP build to the HIP target).
function(add_reference_libraries target backend)
    set(use_hypre FALSE)
    if(HYPRE_FOUND)
        if(backend STREQUAL "HOST" AND NOT HYPRE_WITH_CUDA AND NOT HYPRE_WITH_HIP)
            set(use_hypre TRUE)
        elseif(backend STREQUAL "CUDA" AND HYPRE_WITH_CUDA)
            set(use_hypre TRUE)
        elseif(backend STREQUAL "HIP" AND HYPRE_WITH_HIP)
            set(use_hypre TRUE)
        endif()
    endif()

    if(use_hypre OR Ginkgo_FOUND)
        # MPI must be initialized when using Hypre or Ginkgo
        target_compile_definitions(${target} PUBLIC HAVE_MPI)
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endif()

    if(use_hypre)
        message(STATUS "${target}: using Hypre")
        target_compile_definitions(${target} PUBLIC HAVE_HYPRE)
        target_include_directories(${target} SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS})
        target_link_libraries(${target} PUBLIC ${HYPRE_LIBRARIES} m)
    elseif(HYPRE_FOUND)
        message(STATUS "${target}: Hypre was built for a different backend, benchmarking without Hypre")
    endif()

    if(Ginkgo_FOUND)
        message(STATUS "${target}: using Ginkgo")
        target_compile_definitions(${target} PUBLIC HAVE_GINKGO)
        target_link_libraries(${target} PUBLIC Ginkgo::ginkgo)
    endif()
endfunction()

# host binaries
add_executable(tnl-benchmark-spmv tnl-benchmark-spmv.cpp)
target_link_libraries(tnl-benchmark-spmv PUBLIC TNL::TNL)
tnl_set_host_arch(tnl-benchmark-spmv)

add_executable(tnl-benchmark-spmv-legacy tnl-benchmark-spmv-legacy.cpp)
target_link_libraries(tnl-benchmark-spmv-legacy PUBLIC TNL::TNL)
tnl_set_host_arch(tnl-benchmark-spmv-legacy)

add_executable(tnl-benchmark-spmv-reference tnl-benchmark-spmv-reference.cpp)
target_link_libraries(tnl-benchmark-spmv-reference PUBLIC TNL::TNL)
tnl_set_host_arch(tnl-benchmark-spmv-reference)
add_reference_libraries(tnl-benchmark-spmv-reference HOST)

# separate CUDA binaries
if(BUILD_CUDA)
    add_executable(tnl-benchmark-spmv-cuda tnl-benchmark-spmv.cu)
    target_link_libraries(tnl-benchmark-spmv-cuda PUBLIC TNL::TNL)

    add_executable(tnl-benchmark-spmv-legacy-cuda tnl-benchmark-spmv-legacy.cu)
    target_link_libraries(tnl-benchmark-spmv-legacy-cuda PUBLIC TNL::TNL)

    find_package(CUDAToolkit REQUIRED)
    add_executable(
        tnl-benchmark-spmv-reference-cuda
        tnl-benchmark-spmv-reference.cu
        LightSpMV-1.0/SpMV.cu
        LightSpMV-1.0/SpMVCSR.cu
    )
    target_link_libraries(
        tnl-benchmark-spmv-reference-cuda
        PUBLIC TNL::TNL CUDA::cusparse
    )
    add_reference_libraries(tnl-benchmark-spmv-reference-cuda CUDA)
endif()

# separate HIP binaries
if(BUILD_HIP)
    add_executable(tnl-benchmark-spmv-hip tnl-benchmark-spmv.hip)
    target_link_libraries(tnl-benchmark-spmv-hip PUBLIC TNL::TNL)
    target_compile_definitions(
        tnl-benchmark-spmv-hip
        PUBLIC "-DHIP_ENABLE_PRINTF"
    )

    find_package(hipsparse)
    if(hipsparse_FOUND)
        add_executable(
            tnl-benchmark-spmv-reference-hip
            tnl-benchmark-spmv-reference.hip
        )
        target_link_libraries(
            tnl-benchmark-spmv-reference-hip
            PUBLIC TNL::TNL roc::hipsparse
        )
        add_reference_libraries(tnl-benchmark-spmv-reference-hip HIP)
    else()
        message(
            WARNING
            "hipsparse was not found and so tnl-benchmark-spmv-reference-hip cannot be built."
        )
    endif()
endif()
