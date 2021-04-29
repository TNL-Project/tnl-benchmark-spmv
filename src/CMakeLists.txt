# CSR5 does not work properly yet:
#
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/9
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/10
#
# We can build it with TNL but it crashes with many CUDA errors. We should first check it
# with the original build.
#
#include( cmake/BuildCSR5.cmake )

if( BUILD_CUDA )
    cuda_include_directories( ${CXX_BENCHMARKS_INCLUDE_DIRS} )
    message( STATUS ${CXX_BENCHMARKS_FLAGS} )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cu OPTIONS ${CXX_BENCHMARKS_FLAGS} )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} )
else()
    ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cpp )
    target_compile_options( tnl-benchmark-spmv  PRIVATE ${CXX_BENCHMARKS_FLAGS} )
    target_include_directories( tnl-benchmark-spmv PRIVATE ${CXX_BENCHMARKS_INCLUDE_DIRS} )
endif()

install( TARGETS tnl-benchmark-spmv RUNTIME DESTINATION bin )
