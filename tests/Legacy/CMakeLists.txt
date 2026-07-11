set(COMMON_TESTS
    # TODO: Uncomment the following when AdEllpack works
    #SparseMatrixTest_AdEllpack
    Legacy_SparseMatrixTest_BiEllpack
    Legacy_SparseMatrixTest_ChunkedEllpack
    Legacy_SparseMatrixTest_CSRScalar
    Legacy_SparseMatrixTest_CSRVector
    Legacy_SparseMatrixTest_CSRMultiVector
    Legacy_SparseMatrixTest_CSRLight
    Legacy_SparseMatrixTest_CSRLightWithoutAtomic
    Legacy_SparseMatrixTest_CSRAdaptive
    Legacy_SparseMatrixTest_Ellpack
    Legacy_SparseMatrixTest_SlicedEllpack
)

set(CPP_TESTS ${COMMON_TESTS})
set(CUDA_TESTS ${COMMON_TESTS})

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(
        ${target}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}
    )
endforeach()

foreach(target IN ITEMS ${CUDA_TESTS})
    add_executable(${target}_cuda ${target}.cu)
    set(target ${target}_cuda)
    target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(
        ${target}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}
    )
endforeach()
