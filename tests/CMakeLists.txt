# include CTest only when the project is top-level (not when it is added to
# the build tree of another project with add_subdirectory)
if( PROJECT_IS_TOP_LEVEL )
    # configure the project for testing with CTest/CDash
    include( CTest )

    # repeat failed tests to tolerate sporadic failures
    set(CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --repeat until-pass:2)

    # let CTest write test results in the JUnit XML format
    set(CMAKE_CTEST_ARGUMENTS ${CMAKE_CTEST_ARGUMENTS} --output-junit "${CMAKE_BINARY_DIR}/tests-report.xml")

    if(SYSTEM_GTEST)
        # find GoogleTest installed in the local system
        find_package(GTest REQUIRED)
    else()
        # fetch and build GoogleTest from source
        include(FetchGoogleTest)
    endif()

    set(CXX_TESTS_FLAGS)
    set(CUDA_TESTS_FLAGS)
    set(TESTS_LIBRARIES GTest::gtest_main)
    set(TESTS_LINKER_FLAGS "")
endif()

# add directories with tests
add_subdirectory(Legacy)
