cmake_minimum_required(VERSION 3.24)

# Set CUDA architectures for which the executables will be compiled.
# See https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # NOTE: "all" and "all-major" are available since CMake 3.23, "native" is available since CMake 3.24
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# set cmake's include path so that we can include modules from
# the cmake directory in the TNL repository
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(tnl-benchmark-spmv LANGUAGES CXX)

# declare all custom build options
option( HYPRE_WITH_CUDA "Set if Hypre was built with CUDA on this system." OFF )

# Require C++17, and disable compiler-specific extensions (if possible).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Werror=vla")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# Settings for debug build
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( CMAKE_EXECUTABLE_SUFFIX "-dbg${CMAKE_EXECUTABLE_SUFFIX}" )  # suffix for executables
endif()

# Enforce (more or less) warning-free builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations -Wno-error=uninitialized")

# Warn about redundant semicolons
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi")
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
    )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt")
    endif()
endif()

include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Require C++17, and disable compiler-specific extensions (if possible).
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # set project-specific (i.e. not exported) build options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")

    # enable separable compilation
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)

    # Require C++17, and disable compiler-specific extensions (if possible).
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)

    # Set project-specific (i.e. not exported) build options
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall")
    # NOTE: cmake mirrors the hipcc.pl script and passes some flags to rocm-llvm
    #       (--offload-arch=gfx803 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false)
    #       which do not work with -O0 (leads to Memory access fault), hence -O1
    set(CMAKE_HIP_FLAGS_DEBUG "-O1 -g -DHIP_ENABLE_PRINTF")
    set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELEASE} ${CMAKE_HIP_FLAGS_DEBUG}")
endif()

# Set the project's include path
include_directories(include)

# Add dependencies
if(SYSTEM_TNL)
    find_package(TNL REQUIRED)
else()
    add_subdirectory(libs/tnl EXCLUDE_FROM_ALL)
    set(SYSTEM_TNL TRUE)  # avoid further inclusion of libs/tnl from other libs
endif()

# Add subdirectories with source
add_subdirectory(src)
enable_testing()  # this must be in the top-level CMakeLists.txt
add_subdirectory(tests)
