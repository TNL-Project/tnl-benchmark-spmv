# CSR5 does not work properly yet:
#
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/9
# https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/10
#
# We can build it with TNL but it crashes with many CUDA errors. We should first check it
# with the original build.
#
#include( cmake/BuildCSR5.cmake )

# PETSc requires MPI
#if( TNL_BUILD_MPI AND TNL_USE_PETSC )
#   find_package( PETSc COMPONENTS CXX )
#endif()

#if( PETSC_FOUND )
#   message( "PETSC library found: ${PETSC_VERSION}")
#   set( PETSC_CXX_FLAGS -DHAVE_PETSC ${PETSC_DEFINITIONS} )
#   message( "   PETSC_INCLUDES  = ${PETSC_INCLUDES}" )
#   message( "   PETSC_CXX_FLAGS = ${PETSC_CXX_FLAGS}" )
#   message( "   PETSC_LIBRARIES = ${PETSC_LIBRARIES}" )
#endif()

find_package(MPI COMPONENTS CXX)
if( MPI_CXX_FOUND )
   find_package(Hypre)
   find_package( Ginkgo )
endif()

# host binaries
add_executable( tnl-benchmark-spmv tnl-benchmark-spmv.cpp )
target_link_libraries( tnl-benchmark-spmv PUBLIC TNL::TNL_CXX )

add_executable( tnl-benchmark-spmv-legacy tnl-benchmark-spmv-legacy.cpp )
target_link_libraries( tnl-benchmark-spmv-legacy PUBLIC TNL::TNL_CXX )

if( NOT HYPRE_FOUND OR NOT HYPRE_WITH_CUDA )
   add_executable( tnl-benchmark-spmv-reference tnl-benchmark-spmv-reference.cpp )
   target_link_libraries( tnl-benchmark-spmv-reference PUBLIC TNL::TNL_CXX )
   #target_compile_options( tnl-benchmark-spmv-reference PRIVATE ${PETSC_CXX_FLAGS} )
   #target_include_directories( tnl-benchmark-spmv-reference PRIVATE ${PETSC_INCLUDES} )
   #target_link_libraries( tnl-benchmark-spmv-reference ${PETSC_LIBRARIES} )
else()
   message( "Hypre was built with CUDA on this system and so tnl-benchmark-spmv-reference cannot be built." )
endif()

# separate CUDA binaries
if( CMAKE_CUDA_COMPILER )
   add_executable( tnl-benchmark-spmv-cuda tnl-benchmark-spmv.cu )
   target_link_libraries( tnl-benchmark-spmv-cuda PUBLIC TNL::TNL_CUDA )

   add_executable( tnl-benchmark-spmv-legacy-cuda tnl-benchmark-spmv-legacy.cu )
   target_link_libraries( tnl-benchmark-spmv-legacy-cuda PUBLIC TNL::TNL_CUDA )

   if( NOT HYPRE_FOUND OR HYPRE_WITH_CUDA )
      add_executable( tnl-benchmark-spmv-reference-cuda tnl-benchmark-spmv-reference.cu LightSpMV-1.0/SpMV.cu LightSpMV-1.0/SpMVCSR.cu )
      target_link_libraries( tnl-benchmark-spmv-reference-cuda PUBLIC TNL::TNL_CUDA )
      find_package( CUDAToolkit REQUIRED )
      target_link_libraries( tnl-benchmark-spmv-reference-cuda PUBLIC CUDA::cusparse )
   else()
      message( "Hypre was built without CUDA on this system and so tnl-benchmark-spmv-reference-cuda cannot be built." )
   endif()
endif()

# separate HIP binaries
if( CMAKE_HIP_COMPILER )
   add_executable( tnl-benchmark-spmv-hip tnl-benchmark-spmv.hip )
   target_link_libraries( tnl-benchmark-spmv-hip PUBLIC TNL::TNL_HIP )
   target_compile_definitions( tnl-benchmark-spmv-hip PUBLIC "-DHIP_ENABLE_PRINTF" )

   #find_package( CUDAToolkit REQUIRED ) TODO: Some check for hipsparse
   #target_link_libraries( tnl-benchmark-spmv-reference-hip PUBLIC HIP::hipsparse )

   # Find hipsparse package
   find_package(hipsparse)

   # Check if hipsparse was found
   if(hipsparse_FOUND )
      message(STATUS "hipsparse found")
      if( NOT HYPRE_FOUND OR NOT HYPRE_WITH_CUDA )
         # Link hipsparse to your target (e.g., your executable or library)
         add_executable( tnl-benchmark-spmv-reference-hip tnl-benchmark-spmv-reference.hip )
         target_link_libraries( tnl-benchmark-spmv-reference-hip PUBLIC TNL::TNL_HIP )
         #target_link_libraries( tnl-benchmark-spmv-reference-hip PUBLIC ${hipsparse_LIBRARIES})
         target_link_libraries( tnl-benchmark-spmv-reference-hip PUBLIC roc::hipsparse )
      endif()
   else()
      message(STATUS "hipsparse not found")
      # Handle the case where hipsparse is not found
      # e.g., by providing an error message or fetching the library from a repository
   endif()

endif()

if( MPI_CXX_FOUND )
   if( HYPRE_FOUND )
      if( NOT HYPRE_WITH_CUDA )
         # enable MPI support in TNL
         target_compile_definitions( tnl-benchmark-spmv-reference PUBLIC "-DHAVE_MPI" )
         # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
         target_link_libraries( tnl-benchmark-spmv-reference PUBLIC MPI::MPI_CXX )

         # enable Hypre support in TNL
         target_compile_definitions( tnl-benchmark-spmv-reference PUBLIC "-DHAVE_HYPRE" )
         # add Hypre to the target
         target_include_directories( tnl-benchmark-spmv-reference SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS} )
         target_link_libraries( tnl-benchmark-spmv-reference PUBLIC ${HYPRE_LIBRARIES} -lm )
      endif()

      if( CMAKE_CUDA_COMPILER AND HYPRE_WITH_CUDA )
         # enable MPI support in TNL
         target_compile_definitions( tnl-benchmark-spmv-reference-cuda PUBLIC "-DHAVE_MPI" )
         # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
         target_link_libraries( tnl-benchmark-spmv-reference-cuda PUBLIC MPI::MPI_CXX CUDA::cublas CUDA::curand CUDA::cusparse)

         # enable Hypre support in TNL
         target_compile_definitions( tnl-benchmark-spmv-reference-cuda PUBLIC "-DHAVE_HYPRE" )
         # add Hypre to the target
         target_include_directories( tnl-benchmark-spmv-reference-cuda SYSTEM PUBLIC ${HYPRE_INCLUDE_DIRS} )
         target_link_libraries( tnl-benchmark-spmv-reference-cuda PUBLIC ${HYPRE_LIBRARIES} -lm )
      endif()
   endif()
endif()

if( MPI_CXX_FOUND )
   if( Ginkgo_FOUND )
      if( NOT HYPRE_FOUND OR NOT HYPRE_WITH_CUDA )
         # enable Ginkgo support in TNL
         target_compile_definitions( tnl-benchmark-spmv-reference PUBLIC "-DHAVE_GINKGO -DHAVE_MPI" )
         # add Ginkgo to the target
         target_link_libraries( tnl-benchmark-spmv-reference PUBLIC Ginkgo::ginkgo )
      endif()

      if( CMAKE_CUDA_COMPILER AND ( NOT HYPRE_FOUND OR HYPRE_WITH_CUDA ) )
         # enable Ginkgo support in TNL
         target_compile_definitions( tnl-benchmark-spmv-reference-cuda PUBLIC "-DHAVE_GINKGO -DHAVE_MPI" )
         # add Ginkgo to the target
         target_link_libraries( tnl-benchmark-spmv-reference-cuda PUBLIC Ginkgo::ginkgo )
      endif()
   endif()
endif()