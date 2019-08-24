#http://www.cmake.org/pipermail/cmake/2010-March/036063.html


if ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
   message( WARNING "Code coverage results with an optimised (non-Debug) build may be misleading" )
endif ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   if ( NOT DEFINED CODECOV_OUTPUTFILE )
       set( CODECOV_OUTPUTFILE cmake_coverage.output )
   endif ( NOT DEFINED CODECOV_OUTPUTFILE )

   if ( NOT DEFINED CODECOV_HTMLOUTPUTDIR )
       set( CODECOV_HTMLOUTPUTDIR coverage_results )
   endif ( NOT DEFINED CODECOV_HTMLOUTPUTDIR )

   find_program( CODECOV_GCOV gcov )
   find_program( CODECOV_LCOV lcov )
   find_program( CODECOV_GENHTML genhtml )
   add_compile_options( -fprofile-arcs -ftest-coverage )
   link_libraries( gcov )
   set( CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} --coverage )
   add_custom_target( coverage_init ALL ${CODECOV_LCOV} --base-directory .  --directory ${CMAKE_SOURCE_DIR} --no-external --output-file ${CODECOV_OUTPUTFILE} --capture --initial --quiet )
   add_custom_target( coverage ${CODECOV_LCOV} --base-directory .  --directory ${CMAKE_SOURCE_DIR} --no-external --output-file ${CODECOV_OUTPUTFILE} --capture --quiet COMMAND genhtml --quiet -o ${CODECOV_HTMLOUTPUTDIR} ${CODECOV_OUTPUTFILE} )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   # http://clang.llvm.org/docs/SourceBasedCodeCoverage.html
   add_compile_options( -fprofile-instr-generate -fcoverage-mapping )
   add_link_options( -fprofile-instr-generate -fcoverage-mapping )
   if( ${WITH_CUDA} )
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; -Xcompiler -fprofile-instr-generate ; -Xcompiler -fcoverage-mapping ; -g )
   endif()
endif()
