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
   file( GLOB EXPLICIT_TEMPLATES spmv.templates/*.cu )
   cuda_include_directories( ${CXX_BENCHMARKS_INCLUDE_DIRS} )
   cuda_add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cu ${EXPLICIT_TEMPLATES} ReferenceFormats/LightSpMV-1.0/SpMV.cu ReferenceFormats/LightSpMV-1.0/SpMVCSR.cu
                        OPTIONS ${CXX_BENCHMARKS_FLAGS} ${PETSC_CXX_FLAGS} )
   target_link_libraries( tnl-benchmark-spmv ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} ${PETSC_LINKER_FLAGS})
else()
   file( GLOB EXPLICIT_TEMPLATES spmv.templates/*.cpp )
   add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cpp ${EXPLICIT_TEMPLATES} )
   target_compile_options( tnl-benchmark-spmv  PRIVATE ${CXX_BENCHMARKS_FLAGS} ${PETSC_CXX_FLAGS} )
   target_include_directories( tnl-benchmark-spmv PRIVATE ${CXX_BENCHMARKS_INCLUDE_DIRS} )
   target_link_libraries( tnl-benchmark-spmv ${PETSC_LINKER_FLAGS} )
endif()

install( TARGETS tnl-benchmark-spmv RUNTIME DESTINATION bin )
